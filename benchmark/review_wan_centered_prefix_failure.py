"""Review an interrupted/OOM prefix archive on CPU; never load or rerun Wan."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest


def review(run, root):
    run, root = Path(run), Path(root)
    failure = json.loads((run/'failure.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    progress = json.loads((run/'progress.json').read_text())
    replay = json.loads((run/'replay.json').read_text())
    assert failure['error_type'] == 'OutOfMemoryError', 'This review is specific to the supplied OOM attempt'
    check_values(failure['counts'], {k:v for k,v in progress.items() if k != 'block_entries'})
    check_values(failure['block_entries'], progress['block_entries'])
    check_values(failure['counts'], dict(positive_forwards_attempted=1, positive_forwards_completed=1,
                                      latent_backwards_attempted=1, latent_backwards_completed=0))
    groups = [('pilot', root/'wan_centered_pilot_camel_s1', protocol['inputs_sha256'])]
    for name in ('gradient', 'response', 'destination', 'torso', 'qk', 'shared_input', 'block_input'):
        folder = 'gradients' if name == 'gradient' else name
        groups.append((name, root/f'wan_centered_{folder}_camel_s1', protocol['control_sha256'][name+'_artifacts']))
    groups.append(('source', Path(__file__).parent,
                   {**protocol['helper_sources_sha256'], 'wan_centered_prefix.py':protocol['script_sha256']}))
    provenance = {}
    for group, folder, entries in groups:
        for name, expected in entries.items():
            path = folder/name
            provenance[group+'/'+name] = ('match' if digest(path) == expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    if 'MISMATCH' in provenance.values():
        raise ValueError(provenance)
    control = root/'wan_centered_block_input_camel_s1'
    for filename in ('before_flow.npz', 'before_target_log_probability.npz', 'optimization_support.npz'):
        actual, expected = read_npz(run/filename), read_npz(control/filename)
        assert actual.keys() == expected.keys()
        for name in actual:
            assert np.isfinite(actual[name]).all()
            np.testing.assert_array_equal(actual[name], expected[name])
    mask = read_npz(run/'optimization_support.npz')['common_torso']
    assert mask.dtype == np.bool_ and mask.sum() == 324
    logs = read_npz(run/'before_target_log_probability.npz')
    assert replay['before_flow_exact'] and replay['target_log_probability_exact'] and replay['arms'] == {}
    for arm in ('forward','reverse'):
        np.testing.assert_allclose(-logs[arm][mask].astype(float).mean(), replay['losses'][arm], rtol=1e-6)
    active = {phase:[int(i) for i,n in values.items() if n] for phase,values in progress['block_entries'].items()}
    assert active == {'capture':list(range(21)), 'forward':[19,20], 'reverse':[]}
    assert all(n in (0,1) for values in progress['block_entries'].values() for n in values.values())
    missing = [name for name in ('forward_latent_gradient.npz','reverse_latent_gradient.npz',
        'forward_squared_norms.json','reverse_squared_norms.json','gradient_moments.npz','prefix_report.json')
        if not (run/name).exists()]
    assert len(missing) == 6
    return dict(classification='GPU capacity failure before completing the first backward', failure=failure,
        provenance=provenance, replay=replay, active_blocks=active,
        total_block_entries=sum(sum(v.values()) for v in progress['block_entries'].values()),
        missing_completion_artifacts=missing,
        inference_limits='Exact forward replay does not establish backward parity. No completed latent gradient '
            'or two-arm intermediate cosine exists here. Counters locate recomputation entries, not the allocation site. '
            'This OOM attempt cannot explain the earlier notebook gradient-replay AssertionError.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='probe_runs/wan_centered_prefix_camel_s1')
    parser.add_argument('--root', default='probe_runs')
    args = parser.parse_args()
    metrics = review(args.run, args.root)
    out = Path(args.run)/'review'
    out.mkdir(exist_ok=True)
    (out/'failure_review.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    count = sum(v == 'match' for v in metrics['provenance'].values())
    missing = ', '.join(n for n,v in metrics['provenance'].items() if v != 'match')
    text = f'''# Prefix attempt: out-of-memory review

This supplied attempt failed because the 40 GB GPU ran out of memory during the
first, original-reference backward. It is a different failure from the earlier
notebook's latent-gradient replay AssertionError.

## Recorded allocation failure

- GPU capacity: 39.49 GiB.
- Process memory in use: 39.42 GiB.
- Free memory: 59.44 MiB; requested allocation: 92 MiB.
- PyTorch allocated: 38.16 GiB; reserved but unallocated: 775.94 MiB.

The log establishes memory exhaustion for this attempt, not its unique cause.
It does not establish that fragmentation or one particular tensor is responsible.
More capacity could address OOM but would not by itself resolve gradient replay.

## Work completed

| Work | Completed |
|---|---|
| Original prefix capture, blocks 0-20 | Yes |
| Before-flow and whole target-log-probability replay | Exact |
| Same 324-query support | Exact |
| Forward-reference backward | Started; not completed |
| Reverse-reference backward | Not started |
| Two-reference gradient trace | Unavailable |

There were {metrics['total_block_entries']} recorded block entries: 21 during the
original capture, then recomputation entries for blocks 20 and 19 during the first
backward. This identifies how far recomputation progressed; it does not identify
the specific allocation site or a faulty block.

No complete latent gradient, per-arm squared norms, gradient moments or final prefix
report is present. `replay.json` has an empty `arms` object. We therefore cannot
calculate new gradient agreement, locate alignment in blocks 0/10/15, or evaluate
the earlier replay discrepancy from this archive.

## CPU verification

Failure/progress counters agree. Saved before fields/support match the completed
block-input control exactly; both scalar losses reconstruct from saved log
probabilities. {count} available source/input hashes match. Not supplied locally:
{missing}.

## Next action

Preserve this OOM attempt. Recover the separate gradient-mismatch attempt, if it
still exists: its `replay.json`, `forward_latent_gradient.npz` and
`forward_squared_norms.json` can be analyzed without another GPU run. The two failures
must be kept separate. Do not relax gradient-replay tolerances based on this OOM log.

Before any further full-prefix run, provide more memory headroom or implement and
verify a contained memory reduction with endpoint replay retained. No GPU retry,
memory-strategy change or tolerance change was performed in this review. The existing
evidence does not prove that every 40 GB setup must fail, nor that an 80 GB run would
pass the separate replay gate.
'''
    (out/'REVIEW.md').write_text(text, encoding='utf-8')
    print(out)
    print('Verified hashes:', count, 'Block entries:', metrics['total_block_entries'])


if __name__ == '__main__':
    main()
