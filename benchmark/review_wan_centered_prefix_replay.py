"""CPU analysis of the prefix retry's saved gradient-replay failure."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_gradients import compare


def review(run, root):
    run, root = Path(run), Path(root)
    protocol = json.loads((run/'started.json').read_text())
    failure = json.loads((run/'failure.json').read_text())
    progress = json.loads((run/'progress.json').read_text())
    replay = json.loads((run/'replay.json').read_text())
    norms = json.loads((run/'forward_squared_norms.json').read_text())
    assert failure['error_type'] == 'AssertionError' and 'rtol=0.0005' in failure['error']
    counts = dict(positive_forwards_attempted=1, positive_forwards_completed=1,
                  latent_backwards_attempted=1, latent_backwards_completed=1)
    check_values(failure['counts'], counts)
    check_values({k:v for k,v in progress.items() if k != 'block_entries'}, counts)
    check_values(progress['block_entries'], failure['block_entries'])
    for phase in ('capture', 'forward', 'reverse'):
        assert [progress['block_entries'][phase][str(i)] for i in range(40)] == ([1]*21+[0]*19 if phase != 'reverse' else [0]*40)
    groups = [('pilot', root/'wan_centered_pilot_camel_s1', protocol['inputs_sha256'])]
    for name in ('gradient','response','destination','torso','qk','shared_input','block_input'):
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
    torso = root/'wan_centered_torso_camel_s1'
    for filename in ('before_flow.npz','before_target_log_probability.npz','optimization_support.npz'):
        actual, expected = read_npz(run/filename), read_npz(control/filename)
        assert actual.keys() == expected.keys()
        for n in actual:
            np.testing.assert_array_equal(actual[n], expected[n])
    assert replay['before_flow_exact'] and replay['target_log_probability_exact']
    assert set(replay['arms']) == {'forward'}
    a = read_npz(run/'forward_latent_gradient.npz')['gradient'].astype(float)
    b = read_npz(torso/'forward_latent_gradient.npz')['gradient'].astype(float)
    c = read_npz(torso/'reverse_latent_gradient.npz')['gradient'].astype(float)
    assert a.shape == b.shape == c.shape == (1,16,6,60,104)
    assert all(np.isfinite(v).all() for v in (a,b,c))
    delta = a-b
    comparison = compare(a,b)
    replay_actual = dict(latent_max_abs=float(np.abs(delta).max()), latent_rms=float(np.sqrt((delta**2).mean())),
        latent_cosine=comparison['cosine'], latent_relative_rms=float(np.linalg.norm(delta)/np.linalg.norm(b)))
    check_values(replay_actual, replay['arms']['forward'])
    bad = ~np.isclose(a,b,rtol=5e-4,atol=1e-5)
    assert bad.any() and comparison['cosine'] < .99999
    old_block = read_npz(control/'gradient_moments.npz')['block_input'][:,0]
    np.testing.assert_allclose(norms['block_20'], old_block, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(norms['latent'], (a*a).sum(axis=(0,1,3,4)), rtol=1e-12, atol=1e-12)
    temporal = {}
    for name, energies in norms.items():
        v = np.asarray(energies,float)
        assert v.shape == (6,) and np.isfinite(v).all() and (v>=0).all()
        temporal[name] = dict(norm=float(np.sqrt(v.sum())), energy_fraction_by_frame=(v/v.sum()).tolist())
    old_frames = []
    for i in range(6):
        f,r = b[:,:,i], c[:,:,i]
        old_frames.append(dict(frame=i, comparison=compare(f,r),
            forward_energy_fraction=float((f*f).sum()/(b*b).sum()),
            reverse_energy_fraction=float((r*r).sum()/(c*c).sum())))
    return dict(provenance=provenance, counts=counts, total_block_entries=42,
        same_reference_replay=dict(**replay_actual, angle_degrees=float(np.degrees(np.arccos(comparison['cosine']))),
            new_norm=comparison['forward_norm'], archived_norm=comparison['reverse_norm'],
            norm_ratio=comparison['forward_norm']/comparison['reverse_norm'],
            best_scalar_fit=float((a*b).sum()/(b*b).sum()),
            mismatch_count=int(bad.sum()), element_count=int(bad.size), mismatch_fraction=float(bad.mean())),
        block20_forward_energy_max_abs_difference=float(np.abs(np.asarray(norms['block_20'])-old_block).max()),
        block20_forward_energy_max_relative_difference=float(np.max(np.abs(np.asarray(norms['block_20'])/old_block-1))),
        retry_forward_only_temporal_energy=temporal,
        archived_two_arm_latent=dict(overall=compare(b,c), by_frame=old_frames, excluding_frame0=compare(b[:,:,1:],c[:,:,1:])),
        limits='Same-reference replay cosine is not forward/reverse-reference cosine. Block20 energies alone do not '
               'prove componentwise feature-gradient equality. Only the forward arm ran; no new cross-reference '
               'intermediate cosines can be computed. Temporal energy concentration is not physical motion or a proven cause.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='probe_runs/wan_centered_prefix_camel_s1_retry1')
    parser.add_argument('--root',default='probe_runs')
    args=parser.parse_args()
    m=review(args.run,args.root)
    out=Path(args.run)/'review';out.mkdir(exist_ok=True)
    (out/'replay_review.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    r=m['same_reference_replay']
    rows='\n'.join(f'| {n} | {100*v["energy_fraction_by_frame"][0]:.2f}% |' for n,v in m['retry_forward_only_temporal_energy'].items())
    old=m['archived_two_arm_latent']
    hashes=sum(v=='match' for v in m['provenance'].values())
    text=f'''# Prefix retry 1: gradient-replay failure review

## Failure classification

This attempt did **not** report an out-of-memory error. One prefix forward and the
first (original-reference) backward completed. The latent gradient then failed the
predeclared replay test against the completed torso run. The reverse arm did not run.
There were 42 block entries: 21 capture entries and 21 first-backward recomputation
entries. The two-reference prefix trace is still incomplete.

## Same-reference replay, not a new forward/reverse comparison

| Metric | Measured |
|---|---:|
| New original-reference vs archived original-reference cosine | {r['latent_cosine']:.12f} |
| Angle | {r['angle_degrees']:.6f} degrees |
| Relative RMS difference | {100*r['latent_relative_rms']:.4f}% |
| New / archived norm | {r['new_norm']:.6f} / {r['archived_norm']:.6f} |
| Norm change | {100*(r['norm_ratio']-1):.4f}% |
| Maximum absolute difference | {r['latent_max_abs']:.8f} |
| Elements outside rtol .0005 / atol .00001 | {r['mismatch_count']} / {r['element_count']} ({100*r['mismatch_fraction']:.2f}%) |

The direction is very close to the archive, but both the elementwise gate and the
minimum cosine .99999 gate would fail. The percentage of mismatching entries is not
a percentage of incorrect motion. This is a roughly 1% relative gradient discrepancy,
not evidence that the guidance direction has become wholly different.

Saved block-20 per-frame squared norms replay within about 2e-15 relative error.
This supports continuity at the local endpoint, but matching energies alone does not
prove equality of every gradient component. Current and archived backward paths
differ in instrumentation, and the available records do not establish repeatability
on the same hardware or isolate an instrumentation effect. Ordinary backward numerical
variation, an effect of hooks on execution, or another backward discrepancy remain
possibilities, not established explanations. Do not automatically loosen the gates.

## A secondary clue from already saved arrays

For this retry's **single forward-reference arm**, the fraction of squared gradient
norm in latent frame 0 grows along the backward path:

| Endpoint | Frame-0 share of gradient energy |
|---|---:|
{rows}

These are within-tensor energy fractions, not comparable absolute magnitudes across
variable spaces and not motion measurements. Intermediate values come from saved
per-frame energies, not full feature-gradient arrays. They cannot establish where
the two references become aligned because the reverse arm did not run.

Separately, the completed torso archive already contains both latent gradients.
Its frame-0 forward/reverse cosine is {old['by_frame'][0]['comparison']['cosine']:.6f};
the global value is {old['overall']['cosine']:.6f}. Excluding frame 0 gives
{old['excluding_frame0']['cosine']:.6f}. This supports inspecting temporal concentration
when a valid two-arm trace becomes available; it does not justify masking frame 0
or declaring it the motion-failure cause.

## Verification and next experiment

Reconstructed replay metrics from the full saved latent arrays; verified before
fields/support, counters, current per-frame latent energies and block-20 energy
replay. {hashes} available source/input hashes match; the two large original pilot
step_trace.npz files remain absent locally. No source, tolerance or GPU setting was
changed in this review. The earlier OOM attempt remains a separate record.

Before another two-reference trace, the discriminating control is **same-reference
backward repeatability versus instrumentation** on the same live model/device/state.
A proposed maximum is one prefix capture and three backwards: two without gradient
logging hooks, then one with the logging hooks, all using the original-reference loss.
Account for at most 84 block entries including recomputation; no optimization or
videos. The first two measure ordinary repeat variation; the third tests an added
hook effect. Retain source/runtime/hardware metadata and endpoint replay. This is a
proposal only; it has not been implemented or run. Matching behavior on a tiny CPU
model does not settle pretrained GPU backward repeatability.

If native repeats vary similarly, establish a measured numerical envelope before
revising tolerances explicitly. If hooks cause an additional discrepancy, repair
instrumentation first. If native repeats agree with each other but differ from the
archive, investigate the runtime/backend or archived-control conditions. No new
motion-transfer conclusion is justified by this retry.
'''
    (out/'REVIEW.md').write_text(text,encoding='utf-8')
    print(out)
    print(json.dumps(r,indent=2))
    print('Archived cross-reference cosine excluding frame 0:',old['excluding_frame0']['cosine'])


if __name__=='__main__':
    main()
