"""Reconstruct the saved Q/K diagnostic on CPU; no model or GPU execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_gradients import compare
from benchmark.wan_centered_qk import summarize_moments


def review(run, root):
    run, root = Path(run), Path(root)
    report = json.loads((run/'qk_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    torso = root/'wan_centered_torso_camel_s1'
    groups = [('pilot', root/'wan_centered_pilot_camel_s1', protocol['inputs_sha256'])]
    for name in ('gradient', 'response', 'destination', 'torso'):
        folder = 'gradients' if name == 'gradient' else name
        groups.append((name, root/f'wan_centered_{folder}_camel_s1',
                       protocol['control_sha256'][name+'_artifacts']))
    groups.append(('source', Path(__file__).parent,
                   {**protocol['helper_sources_sha256'], 'wan_centered_qk.py':protocol['script_sha256']}))
    provenance = {}
    for group, folder, entries in groups:
        for name, expected in entries.items():
            path = folder/name
            provenance[group+'/'+name] = ('match' if digest(path) == expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    if 'MISMATCH' in provenance.values():
        raise ValueError(provenance)
    counts = dict(positive_forwards_attempted=1, positive_forwards_completed=1,
                  readout_backwards_attempted=2, readout_backwards_completed=2)
    check_values(report['counts'], counts)
    check_values(json.loads((run/'progress.json').read_text()), counts)
    assert report['baseline_unchanged'] and not report['latent_graph_created'] and not report['port_success']
    assert report['qk_dtype'] == 'torch.bfloat16'
    for name in ('before_flow.npz', 'before_target_log_probability.npz'):
        actual, expected = read_npz(run/name), read_npz(torso/name)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert np.isfinite(actual[key]).all()
            np.testing.assert_array_equal(actual[key], expected[key])
    support = read_npz(run/'optimization_support.npz')['common_torso']
    np.testing.assert_array_equal(support, read_npz(torso/'optimization_support.npz')['common_torso'])
    assert support.dtype == np.bool_ and support.sum() == 324
    assert report['flow_replay_max_abs'] == 0
    logp = read_npz(run/'before_target_log_probability.npz')
    for arm in ('forward', 'reverse'):
        assert report['replay'][arm]['log_probability_max_abs'] == 0
        np.testing.assert_allclose(-logp[arm][support].astype(float).mean(),
                                   report['replay'][arm]['loss'], rtol=1e-6)
    moments = read_npz(run/'gradient_moments.npz')
    assert set(moments) == {'Q', 'K', 'centered_scores', 'raw_scores'}
    for name, v in moments.items():
        assert v.dtype == np.float64 and np.isfinite(v).all()
        assert v.shape == ((6, 40, 3) if name in ('Q', 'K') else (5, 3))
        aa, bb, ab = np.moveaxis(v, -1, 0)
        assert (aa >= 0).all() and (bb >= 0).all()
        assert (ab**2 <= aa*bb*(1+1e-10)+1e-30).all()
    np.testing.assert_array_equal(moments['Q'][-1], 0)
    np.testing.assert_array_equal(moments['K'][0], 0)
    stages = {name:summarize_moments(v) for name, v in moments.items()}
    stages['joint_QK'] = summarize_moments(np.concatenate([moments['Q'], moments['K']]))
    check_values(stages, report['stage_comparison'])
    by_frame = {name:[dict(frame=i, **summarize_moments(v)) for i, v in enumerate(moments[name])]
                for name in ('Q', 'K')}
    by_head = {name:[dict(head=i, **summarize_moments(moments[name][:, i])) for i in range(40)]
               for name in ('Q', 'K')}
    by_pair = {name:[dict(pair=[i, i+1], **summarize_moments(v)) for i, v in enumerate(moments[name])]
               for name in ('centered_scores', 'raw_scores')}
    check_values(by_frame, report['by_frame'])
    check_values(by_head, report['by_head'])
    check_values(by_pair, report['score_by_pair'])
    gradients = [read_npz(torso/f'{a}_latent_gradient.npz')['gradient'] for a in ('forward', 'reverse')]
    assert all(np.isfinite(v).all() for v in gradients)
    latent = compare(*gradients)
    check_values(latent, report['archived_latent_comparison'])
    head_summary = {}
    for name in ('Q', 'K'):
        energy = moments[name][:, :, :2].sum(axis=(0, 2))
        cosines = np.array([v['cosine'] for v in by_head[name]])
        head_summary[name] = dict(cosine_min=float(cosines.min()), cosine_median=float(np.median(cosines)),
            cosine_max=float(cosines.max()), largest_head_energy_fraction=float(energy.max()/energy.sum()))
    return dict(provenance=provenance, counts=counts, stage_comparison=stages,
        archived_latent_comparison=latent, by_frame=by_frame, by_head=by_head, score_by_pair=by_pair,
        head_summary=head_summary, replay='Before flow and both target log-probability fields exactly match torso control.',
        limits='Q/K comparisons reconstructed from saved norm/dot sufficient statistics, not full gradients. '
               'Nonzero counts and maxima remain GPU-reported. Latent comparison reconstructed from full saved gradients. '
               'Different coordinate spaces; this localizes alignment but does not prove a bug or a decoded-motion cause.')


def write_review(run, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(run)/'review'
    out.mkdir(exist_ok=True)
    (out/'review_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    order = ['centered_scores', 'raw_scores', 'Q', 'K', 'joint_QK']
    values = [metrics['stage_comparison'][n]['cosine'] for n in order]
    values.append(metrics['archived_latent_comparison']['cosine'])
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    bars = ax.bar(['Centered\nscores', 'Raw\nscores', 'Q', 'K', 'Joint Q/K', 'Noisy latent\n(archived)'],
                  values, color=['#427aa1']*5+['#b94e48'])
    ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set(ylim=(0, 1.09), ylabel='Forward / reverse gradient cosine',
           title='Reference gradients remain distinct at Q/K; align on the path back to the latent')
    ax.text(.01, .97, '1 = same direction; 0 = orthogonal. Coordinate-dependent diagnostic, not motion transfer.',
            transform=ax.transAxes, va='top', fontsize=9)
    fig.savefig(out/'qk_gradient_stages.png', dpi=160)
    plt.close(fig)
    rows = '\n'.join(f'| {n.replace("_", " ")} | {metrics["stage_comparison"][n]["cosine"]:.6f} |' for n in order)
    rows += f'\n| Noisy latent (archived) | {values[-1]:.6f} |'
    matched = sum(v == 'match' for v in metrics['provenance'].values())
    missing = [k for k, v in metrics['provenance'].items() if v != 'match']
    heads = '\n'.join(f'- {n}: head cosine range {v["cosine_min"]:.3f} to {v["cosine_max"]:.3f}; '
                      f'median {v["cosine_median"]:.3f}; largest head {100*v["largest_head_energy_fraction"]:.1f}% '
                      'of combined forward/reverse squared gradient norm.' for n, v in metrics['head_summary'].items())
    text = f'''# Completed Q/K gradient-stage review

## Finding

The hypothesis that score-to-Q/K differentiation already makes the two reference
gradients nearly parallel is not supported at this saved state. They remain distinct
at native normalized, post-RoPE Q/K. Near-alignment appears in the remaining backward
mapping from Q/K to the noisy latent, which includes the Q/K branches merging into a
shared attention input, normalization/projections, and preceding transformer blocks.
This identifies an interval, not the responsible operation or a software defect.

| Differentiated variable | Forward/reverse cosine |
|---|---:|
{rows}

![Gradient directions](qk_gradient_stages.png)

Centering barely changes the cosine (.469136 to .469172). Q and K are both distinct;
the joint result is not concealing an almost-aligned component. Native BF16 readout
backward preserves this distinction. Precision elsewhere remains untested by this
experiment. Norms in different variable spaces cannot establish vanishing gradients.

{heads}

## Evidence checks and scope

- Completed exactly one positive capture and two detached readout backwards.
- Zero latent backwards, optimizer updates, scheduler steps or video generations.
- Before flow and both whole target-log-probability fields replay exactly.
- Same 324 common torso queries; unchanged before latent reported by the GPU runner.
- Reconstructed every stage, frame, head and score-pair comparison from finite FP64
  norm/dot statistics; checked nonnegative energies and Cauchy bounds.
- Last-frame Q and first-frame K moments are exactly zero, as expected for adjacent pairs.
- Reconstructed archived latent comparison from the two complete saved gradient arrays.
- {matched} available source/input hashes match. Not supplied locally: {', '.join(missing)}.
- Full Q/K feature gradients were intentionally not exported. Their entry counts and
  maxima are GPU-reported; sufficient statistics independently reproduce the reported
  comparisons, not the original autograd execution.

## Why the reversed arm can worsen

The arms start from the same noisy latent and therefore the same estimated flow,
but use independent updates and different fixed reference destinations. Neither
arm receives the other's updates. Reversing the reference also need not give the
exact negative displacement at each identical spatial query.

The torso test optimizes destination negative log-probability, not expected-coordinate
error. Increasing probability at the requested destination can coexist with a worse
probability-weighted coordinate when mass elsewhere moves. Both arms lowering that
loss therefore does not guarantee that both coordinate errors improve. Their nearly
parallel latent gradients also show weak reference selectivity at the starting state;
this does not by itself prove why every later metric worsens.

## Next discriminating measurement

Measure the two gradients at the **shared input to block 20's Q/K projections**, with
the same state, targets, support and objective. Compare with these Q/K and latent
results. If alignment is already high there, investigate the local Q/K projection,
normalization and branch merge. If it remains low, investigate the preceding
transformer path. This is a proposed localization test, not an implemented repair;
retain a bounded capture/backward budget and replay gates when implementing it.

No guidance has been repaired or decoded motion transfer demonstrated by this run.
The last decoded comparison remains unsuccessful.
'''
    (out/'REVIEW.md').write_text(text, encoding='utf-8')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='probe_runs/wan_centered_qk_camel_s1')
    parser.add_argument('--root', default='probe_runs')
    args = parser.parse_args()
    metrics = review(args.run, args.root)
    print(write_review(args.run, metrics))
    print(json.dumps(metrics['head_summary'], indent=2))


if __name__ == '__main__':
    main()
