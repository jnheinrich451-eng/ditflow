"""CPU review of the shared-input probe; reconstruct saved statistics, no GPU work."""
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
    report = json.loads((run/'shared_input_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    qk = root/'wan_centered_qk_camel_s1'
    torso = root/'wan_centered_torso_camel_s1'
    groups = [('pilot', root/'wan_centered_pilot_camel_s1', protocol['inputs_sha256'])]
    for name in ('gradient', 'response', 'destination', 'torso', 'qk'):
        folder = 'gradients' if name == 'gradient' else name
        groups.append((name, root/f'wan_centered_{folder}_camel_s1', protocol['control_sha256'][name+'_artifacts']))
    groups.append(('source', Path(__file__).parent,
        {**protocol['helper_sources_sha256'], 'wan_centered_shared_input.py':protocol['script_sha256']}))
    provenance = {}
    for group, folder, entries in groups:
        for name, expected in entries.items():
            path = folder/name
            provenance[group+'/'+name] = ('match' if digest(path) == expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    if 'MISMATCH' in provenance.values():
        raise ValueError(provenance)
    expected_counts = dict(positive_forwards_attempted=1, positive_forwards_completed=1,
                           local_backwards_attempted=2, local_backwards_completed=2)
    check_values(report['counts'], expected_counts)
    check_values(json.loads((run/'progress.json').read_text()), expected_counts)
    assert report['baseline_unchanged'] and not report['latent_graph_created'] and not report['port_success']
    assert report['shared_input_dtype'] == 'torch.bfloat16' and report['shared_input_shape'] == [1, 9360, 5120]
    assert all(report['replay'][n] for n in ('native_qk_exact', 'before_flow_exact', 'target_log_probability_exact'))
    check_values(json.loads((run/'replay.json').read_text()), report['replay'])
    for name in ('before_flow.npz', 'before_target_log_probability.npz', 'optimization_support.npz'):
        actual, expected = read_npz(run/name), read_npz(qk/name)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert np.isfinite(actual[key]).all()
            np.testing.assert_array_equal(actual[key], expected[key])
    support = read_npz(run/'optimization_support.npz')['common_torso']
    assert support.dtype == np.bool_ and support.sum() == 324
    logs = read_npz(run/'before_target_log_probability.npz')
    for arm in ('forward', 'reverse'):
        np.testing.assert_allclose(-logs[arm][support].astype(float).mean(), report['replay']['losses'][arm], rtol=1e-6)
        assert report['branch_sum'][arm]['native_sum_exact']
    moments = read_npz(run/'gradient_moments.npz')
    names = ('Q', 'K', 'input_Q_path', 'input_K_path', 'input_combined')
    assert set(moments) == set(names) | {'forward_within_Q_K', 'reverse_within_Q_K'}
    for name, v in moments.items():
        assert v.dtype == np.float64 and np.isfinite(v).all()
        assert v.shape == ((6, 40, 3) if name in ('Q', 'K') else (6, 3))
        aa, bb, ab = np.moveaxis(v, -1, 0)
        assert (aa >= 0).all() and (bb >= 0).all()
        assert (ab**2 <= aa*bb*(1+1e-10)+1e-30).all()
    old = read_npz(qk/'gradient_moments.npz')
    for name in ('Q', 'K'):
        np.testing.assert_array_equal(moments[name], old[name])
    for name, frame in [('Q', -1), ('K', 0), ('input_Q_path', -1), ('input_K_path', 0)]:
        np.testing.assert_array_equal(moments[name][frame], 0)
    stages = {n:summarize_moments(moments[n]) for n in names}
    check_values(stages, report['stage_comparison'])
    frames = {n:[dict(frame=i, **summarize_moments(v)) for i, v in enumerate(moments[n])] for n in names}
    check_values(frames, report['by_frame'])
    branches, rounding = {}, {}
    for i, arm in enumerate(('forward', 'reverse')):
        within = moments[arm+'_within_Q_K']
        # Within-arm statistics must recover the same branch energies per frame.
        np.testing.assert_array_equal(within[:, 0], moments['input_Q_path'][:, i])
        np.testing.assert_array_equal(within[:, 1], moments['input_K_path'][:, i])
        v = summarize_moments(within)
        native_norm = stages['input_combined'][arm+'_norm']
        branches[arm] = dict(q_path_norm=v['forward_norm'], k_path_norm=v['reverse_norm'], cosine=v['cosine'],
            sum_norm_before_native_rounding=2*v['shared_mean_norm'], native_sum_norm=native_norm,
            native_sum_to_path_norms_ratio=native_norm/(v['forward_norm']+v['reverse_norm']))
        rounding[arm] = native_norm/branches[arm]['sum_norm_before_native_rounding']-1
    check_values(branches, report['within_arm_branch_comparison'])
    qk_report = json.loads((qk/'qk_report.json').read_text())
    check_values(qk_report['stage_comparison'], report['archived_qk_comparison'])
    latent = compare(*[read_npz(torso/f'{a}_latent_gradient.npz')['gradient'] for a in ('forward', 'reverse')])
    check_values(latent, report['archived_latent_comparison'])
    combined = frames['input_combined']
    return dict(provenance=provenance, counts=expected_counts, stage_comparison=stages,
        archived_qk_comparison=qk_report['stage_comparison'], archived_latent_comparison=latent,
        by_frame=frames, within_arm_branch_comparison=branches,
        native_sum_norm_relative_rounding_change=rounding,
        combined_frame_cosine_range=[min(v['cosine'] for v in combined), max(v['cosine'] for v in combined)],
        limits='Feature-gradient comparisons are reconstructed from saved sufficient statistics, not full gradients. '
               'Native Q/K equality and elementwise branch-sum equality remain GPU-reported; saved fields and Q/K moments '
               'independently replay exactly. Norm rounding change does not bound angular rounding error. '
               'This localizes alignment in this objective/state, not a decoded-motion cause or repair.')


def write_review(run, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(run)/'review'
    output.mkdir(exist_ok=True)
    (output/'review_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    names = ('Q', 'K', 'input_Q_path', 'input_K_path', 'input_combined')
    values = [metrics['stage_comparison'][n]['cosine'] for n in names]
    values.append(metrics['archived_latent_comparison']['cosine'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), width_ratios=[1.5, 1], layout='constrained')
    bars = axes[0].bar(['Q', 'K', 'Input via Q', 'Input via K', 'Input sum', 'Latent\n(archived)'], values,
                       color=['#447da8']*4+['#328267', '#b44e43'])
    axes[0].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[0].set(ylim=(0, 1.08), ylabel='Forward / reverse gradient cosine', title='Reference directions remain distinct at the shared input')
    axes[0].tick_params(axis='x', labelsize=8)
    for name, label in [('input_Q_path', 'Via Q'), ('input_K_path', 'Via K'), ('input_combined', 'Combined')]:
        rows = metrics['by_frame'][name]
        axes[1].plot([v['frame'] for v in rows], [np.nan if v['cosine'] is None else v['cosine'] for v in rows], 'o-', label=label)
    axes[1].axhline(values[-1], color='#b44e43', linestyle='--', label='Latent overall (archived)')
    axes[1].set(ylim=(0, 1.08), xlabel='Latent-frame index', ylabel='Forward / reverse cosine', title='No near-aligned shared-input frame')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'shared_input_gradients.png', dpi=160)
    plt.close(fig)
    rows = '\n'.join(f'| {n.replace("_", " ")} | {metrics["stage_comparison"][n]["cosine"]:.6f} |' for n in names)
    branches = metrics['within_arm_branch_comparison']
    matched = sum(v == 'match' for v in metrics['provenance'].values())
    missing = ', '.join(n for n, v in metrics['provenance'].items() if v != 'match')
    lo, hi = metrics['combined_frame_cosine_range']
    text = f'''# Completed shared-input probe review

## Finding

Local Q/K projection, normalization and their gradient combination do not produce
the observed near-alignment at this saved state. Forward/reverse gradient directions
remain distinct at the shared input to block 20's self-attention. The hypothesis
that this local mapping already explains the .975 latent cosine is not supported.

| Differentiated variable | Forward/reverse cosine |
|---|---:|
{rows}
| Noisy latent (archived) | {values[-1]:.6f} |

![Gradient comparison](shared_input_gradients.png)

Combined-input cosine is {lo:.3f} to {hi:.3f} across all six latent frames. Thus the
overall result does not conceal a near-aligned individual frame. Q/K moments exactly
match the completed Q/K run.

## Branch combination

Within the original-reference arm, Q-path versus K-path cosine is
{branches['forward']['cosine']:.6f}; within the reversed-reference arm it is
{branches['reverse']['cosine']:.6f}. These compare Q and K contributions **within**
each arm, unlike the forward/reverse table above. They are approximately orthogonal,
not strongly opposed. There is no evidence of severe aggregate Q/K cancellation.

Combined-input gradient norms are {branches['forward']['native_sum_norm']:.6f} and
{branches['reverse']['native_sum_norm']:.6f}. The sum-to-path-norms ratios are
{branches['forward']['native_sum_to_path_norms_ratio']:.6f} and
{branches['reverse']['native_sum_to_path_norms_ratio']:.6f}; a ratio around .707 is
expected for equal-norm perpendicular vectors, so it does not mean 29% of the
gradient was erroneously lost. Cross-space norm comparisons cannot diagnose
vanishing gradients.

Native BF16 summation changes the combined norms by less than .002% relative to
the sums derived from FP64 norm/dot statistics. This norm comparison alone is not
an angular-error bound. The GPU runner reports exact equality of the native-dtype
elementwise branch sum and shared-input gradient.

## Evidence checks

- Exactly one native forward and two local backwards; no latent graph, optimization,
  scheduler step or video generation.
- Saved flow, both whole target-log-probability fields, and the 324-query support
  replay exactly against the completed Q/K control.
- Both Q/K per-frame/head moment arrays are exactly equal to the prior run.
- Reconstructed every stage/frame comparison and both within-arm comparisons from
  finite FP64 moments. Checked nonnegative energies and Cauchy bounds.
- Within-arm branch energies agree with the cross-arm branch statistics per frame.
- Expected unused last-Q/first-K frames have zero feature and input-branch moments.
- Archived latent comparison reconstructed from complete saved latent gradients.
- {matched} available source/input hashes match. Not supplied locally: {missing}.
- Full feature gradients were not exported. Native Q/K replay and elementwise
  branch-sum equality remain GPU-reported checks; the CPU review independently
  reconstructs statistics and compares the saved arrays.

## What remains unresolved

The remaining interval lies **before this attention input in the forward pass**:
block 20's pre-attention normalization/time modulation and the preceding transformer
path back to the latent. The near-alignment could arise gradually or at one operation.
Cosines in different variable spaces locate a change but do not by themselves prove
a software defect, inaccurate AMF correspondences, or the cause of decoded failure.

Monitoring Q/K has helped eliminate the local similarity-to-Q/K and Q/K-to-shared-
input mappings as the points where near-alignment is already present. It does not
validate the semantic matches or repair motion transfer. There is no justification
from this result alone for changing heads, detaching a branch, or raising learning rate.

## Next discriminating test (proposal only)

Measure the two gradients at block 20's incoming residual features, before its
pre-attention normalization and time modulation. Keep the loss, references, support,
state and model fixed, with replay of the current shared-input gradients. A local
capture with two backwards stopping at those detached incoming features can
separate this final local mapping from the earlier transformer stack. State and
enforce that budget before execution; no new GPU experiment was run in this review.

If near-alignment appears there, inspect the normalization/modulation mapping. If
the gradients remain distinct, the missing change lies earlier in the transformer.
The last decoded comparison remains unsuccessful; this probe provides localization,
not a motion-transfer result.
'''
    (output/'REVIEW.md').write_text(text, encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='probe_runs/wan_centered_shared_input_camel_s1')
    parser.add_argument('--root', default='probe_runs')
    args = parser.parse_args()
    metrics = review(args.run, args.root)
    print(write_review(args.run, metrics))
    print('Combined frame cosine range:', metrics['combined_frame_cosine_range'])
    print('Relative norm rounding changes:', metrics['native_sum_norm_relative_rounding_change'])


if __name__ == '__main__':
    main()
