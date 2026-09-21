"""CPU review of block-input gradient moments; no model execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_gradients import compare
from benchmark.wan_centered_qk import summarize_moments


STAGES = ('Q', 'K', 'attention_input', 'modulated_fp32', 'normalized_fp32', 'norm_input_fp32', 'block_input')


def review(run, root):
    run, root = Path(run), Path(root)
    report = json.loads((run/'block_input_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    shared = root/'wan_centered_shared_input_camel_s1'
    torso = root/'wan_centered_torso_camel_s1'
    groups = [('pilot', root/'wan_centered_pilot_camel_s1', protocol['inputs_sha256'])]
    for name in ('gradient', 'response', 'destination', 'torso', 'qk', 'shared_input'):
        folder = 'gradients' if name == 'gradient' else name
        groups.append((name, root/f'wan_centered_{folder}_camel_s1', protocol['control_sha256'][name+'_artifacts']))
    groups.append(('source', Path(__file__).parent,
        {**protocol['helper_sources_sha256'], 'wan_centered_block_input.py':protocol['script_sha256']}))
    provenance = {}
    for group, folder, entries in groups:
        for name, expected in entries.items():
            path = folder/name
            provenance[group+'/'+name] = ('match' if digest(path) == expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    if 'MISMATCH' in provenance.values():
        raise ValueError(provenance)
    counts = dict(positive_forwards_attempted=1, positive_forwards_completed=1,
                  local_backwards_attempted=2, local_backwards_completed=2)
    check_values(report['counts'], counts)
    check_values(json.loads((run/'progress.json').read_text()), counts)
    assert report['baseline_unchanged'] and not report['latent_graph_created'] and not report['port_success']
    assert report['block_input_shape'] == [1, 9360, 5120]
    check_values(report['endpoint_dtypes'], {n:'torch.bfloat16' if n in ('Q','K','attention_input','block_input') else 'torch.float32' for n in STAGES})
    check_values(json.loads((run/'replay.json').read_text()), report['replay'])
    assert all(report['replay'][n] for n in ('attention_input_exact', 'native_qk_exact', 'before_flow_exact', 'target_log_probability_exact'))
    for name in ('before_flow.npz', 'before_target_log_probability.npz', 'optimization_support.npz'):
        actual, expected = read_npz(run/name), read_npz(shared/name)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert np.isfinite(actual[key]).all()
            np.testing.assert_array_equal(actual[key], expected[key])
    support = read_npz(run/'optimization_support.npz')['common_torso']
    assert support.dtype == np.bool_ and support.sum() == 324
    logs = read_npz(run/'before_target_log_probability.npz')
    for a in ('forward', 'reverse'):
        np.testing.assert_allclose(-logs[a][support].astype(float).mean(), report['replay']['losses'][a], rtol=1e-6)
        check_values(report['derivative_identity_checks'][a], dict(attention_cast_exact=True, modulation_exact=True, input_cast_exact=True))
    moments = read_npz(run/'gradient_moments.npz')
    assert set(moments) == set(STAGES)
    for name, v in moments.items():
        assert v.dtype == np.float64 and np.isfinite(v).all()
        assert v.shape == ((6, 40, 3) if name in ('Q', 'K') else (6, 3))
        aa, bb, ab = np.moveaxis(v, -1, 0)
        assert (aa >= 0).all() and (bb >= 0).all()
        assert (ab**2 <= aa*bb*(1+1e-10)+1e-30).all()
    old = read_npz(shared/'gradient_moments.npz')
    for name, previous in [('Q', 'Q'), ('K', 'K'), ('attention_input', 'input_combined')]:
        np.testing.assert_array_equal(moments[name], old[previous])
    np.testing.assert_array_equal(moments['attention_input'], moments['modulated_fp32'])
    np.testing.assert_array_equal(moments['Q'][-1], 0)
    np.testing.assert_array_equal(moments['K'][0], 0)
    stages = {n:summarize_moments(moments[n]) for n in STAGES}
    frames = {n:[dict(frame=i, **summarize_moments(v)) for i, v in enumerate(moments[n])] for n in STAGES}
    check_values(stages, report['stage_comparison'])
    check_values(frames, report['by_frame'])
    control = json.loads((shared/'shared_input_report.json').read_text())
    check_values(control['stage_comparison'], report['archived_shared_comparison'])
    latent = compare(*[read_npz(torso/f'{a}_latent_gradient.npz')['gradient'] for a in ('forward', 'reverse')])
    check_values(latent, report['archived_latent_comparison'])
    sequence = list(STAGES[2:])
    transitions = {f'{a}_to_{b}':stages[b]['cosine']-stages[a]['cosine'] for a, b in zip(sequence, sequence[1:])}
    return dict(provenance=provenance, counts=counts, stage_comparison=stages, by_frame=frames,
        archived_latent_comparison=latent, cosine_changes_backward=transitions,
        block_input_frame_cosine_range=[min(v['cosine'] for v in frames['block_input']), max(v['cosine'] for v in frames['block_input'])],
        limits='Comparisons reconstructed from norm/dot sufficient statistics, not full feature gradients. Native feature '
               'replay and elementwise derivative identities remain GPU-reported. Saved flow, log probabilities, Q/K and '
               'attention-input moments independently replay exactly. Different coordinate spaces; localization is not a '
               'bug diagnosis or decoded-motion result. Latent gradients come from the completed earlier run.')


def write_review(run, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(run)/'review'
    out.mkdir(exist_ok=True)
    (out/'review_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    order = list(STAGES[2:])
    values = [metrics['stage_comparison'][n]['cosine'] for n in order]
    latent = metrics['archived_latent_comparison']['cosine']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), width_ratios=[1.5, 1], layout='constrained')
    bars = axes[0].bar(['Attention\ninput', 'After output\ncast backward', 'After scale\nbackward', 'After norm\nbackward', 'Block 20\ninput', 'Latent\n(archived)'],
                       values+[latent], color=['#427ea5']*4+['#35856e','#b84e43'])
    axes[0].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[0].set(ylim=(0,1.08), ylabel='Forward / reverse gradient cosine', title='No local operation produces near-alignment')
    axes[0].tick_params(axis='x', labelsize=8)
    for name, label in [('attention_input','Attention input'),('normalized_fp32','After scale backward'),('block_input','Block input')]:
        rows = metrics['by_frame'][name]
        axes[1].plot([v['frame'] for v in rows], [v['cosine'] for v in rows], 'o-', label=label)
    axes[1].axhline(latent, color='#b84e43', linestyle='--', label='Latent overall (archived)')
    axes[1].set(ylim=(0,1.08), xlabel='Latent-frame index', ylabel='Forward / reverse cosine', title='Directions remain distinct in all six frames')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'block_input_gradients.png', dpi=160)
    plt.close(fig)
    rows = '\n'.join(f'| {n.replace("_", " ")} | {metrics["stage_comparison"][n]["cosine"]:.9f} |' for n in order)
    matched = sum(v == 'match' for v in metrics['provenance'].values())
    missing = ', '.join(n for n, v in metrics['provenance'].items() if v != 'match')
    lo, hi = metrics['block_input_frame_cosine_range']
    text = f'''# Completed block 20 input probe review

## Finding

The two references' gradient directions remain distinct all the way back to block
20's incoming features. The local pre-attention path does not produce the observed
near-alignment at the noisy latent. The specific hypothesis tested here is not supported.

Read the table in backward order, toward the latent:

| Gradient endpoint | Forward/reverse cosine |
|---|---:|
{rows}
| Noisy latent (archived) | {latent:.9f} |

![Local backward path](block_input_gradients.png)

The output-cast backward preserves the gradient moments exactly. Backward through
timestep scaling reduces cosine from .313218 to .292395. LayerNorm changes it to
.291575, and the incoming-feature cast changes it by less than .000001. These local
casts do not explain the .975049 latent result; this does not exonerate precision
throughout the whole model. Local block-input norms are .925028/.874404, but norms
across different variable spaces cannot diagnose gradient vanishing or amplification.

Across the six latent frames, block-input cosine ranges from {lo:.3f} to {hi:.3f}.
The aggregate therefore does not conceal a near-aligned individual frame.

## What this resolves

For the fixed objective/state, near-alignment has not yet appeared after traversing
the score-to-Q/K mapping, local Q/K projections and normalization, branch combination,
or block 20's pre-attention normalization/time modulation. The unresolved interval
is now the forward mapping from the noisy latent through input casting, patch
embedding and blocks 0-19 to the incoming features of block 20.

This is useful localization, not a causal explanation of decoded failure. A gradual
change through several blocks remains possible. No particular earlier block, attention
head, correspondence rule, or precision setting has been identified as faulty.
Block 20 remains an unvalidated motion-guidance choice. Semantic correspondence
quality and reference-dependent decoded motion are separate unresolved requirements.

## Verification

- Exactly one native forward and two local backwards; no latent graph, updates,
  scheduler steps or videos.
- Whole saved flow/log-probability fields and the same 324-query support replay exactly.
- Q/K and shared-attention-input moments are exactly equal to the prior control.
- Reconstructed all seven stage summaries and all per-frame comparisons from finite
  FP64 norm/dot statistics; checked energies and Cauchy bounds.
- Reconstructed archived latent comparison from complete saved latent gradients.
- The GPU reports exact native feature replay and all cast/modulation derivative
  identities. Full feature gradients were not exported, so those elementwise checks
  cannot be independently repeated from the compact archive.
- {matched} available source/input hashes match. Not supplied locally: {missing}.
- The current experiment has no latent backward. The .975049 endpoint is reused
  evidence from the completed torso run, not a new measurement in this graph.

## Next discriminating experiment: one coarse trace of the earlier path

Use the same saved state and two reference objectives. In one gradient-enabled
prefix evaluation, measure gradients at block inputs **20, 10 and 0**, plus the noisy
latent, using two objective backwards. Replay block-20 and latent endpoints against
the completed controls. These are logging checkpoints on the existing block-20
objective, not three different guidance layers or new generation conditions.

- A change between inputs 20 and 10 implicates the mapping through blocks 10-19.
- A change between inputs 10 and 0 implicates the mapping through blocks 0-9.
- A change between block input 0 and the latent implicates patch embedding/input casting.
- Similar intermediate values would instead show gradual alignment, and an endpoint
  replay discrepancy must be resolved before any localization claim.

This is a proposal, not an implemented or executed GPU test. It requires full-prefix
backwards and is more expensive than the local probes. Set an explicit budget and
count checkpoint recomputation separately when implementing it; two backwards do not
mean only two physical forward executions. No optimizer steps or decoded videos are
needed for this localization. Do not continue with one GPU run per individual block.

The last decoded comparison remains unsuccessful. None of these gradient comparisons
establishes a repaired port or physical motion transfer.
'''
    (out/'REVIEW.md').write_text(text, encoding='utf-8')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='probe_runs/wan_centered_block_input_camel_s1')
    parser.add_argument('--root', default='probe_runs')
    args = parser.parse_args()
    metrics = review(args.run, args.root)
    print(write_review(args.run, metrics))
    print('Frame cosine range:', metrics['block_input_frame_cosine_range'])
    print('Local backward cosine changes:', metrics['cosine_changes_backward'])


if __name__ == '__main__':
    main()
