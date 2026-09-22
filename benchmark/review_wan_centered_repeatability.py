"""Recompute the saved repeatability evidence on CPU; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def metrics(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    assert na > 0 and nb > 0
    cosine = float(np.sum(a*b)/(na*nb))
    return dict(cosine=cosine, actual_norm=float(na), reference_norm=float(nb),
        max_abs=float(np.abs(a-b).max()), rms=float(np.sqrt(np.mean((a-b)**2))),
        relative_rms=float(np.linalg.norm(a-b)/nb),
        old_elementwise_gate_passed=bool(np.allclose(a,b,rtol=5e-4,atol=1e-5)),
        old_direction_gate_passed=bool(cosine >= .99999))


def review(run, root):
    run, root = Path(run), Path(root)
    protocol, report = read(run/'started.json'), read(run/'repeatability_report.json')
    progress, replay = read(run/'progress.json'), read(run/'replay.json')
    assert not (run/'failure.json').exists()
    counts = dict(positive_forwards_attempted=1,positive_forwards_completed=1,
        latent_backwards_attempted=3,latent_backwards_completed=3)
    assert report['counts'] == counts == {k:v for k,v in progress.items() if k != 'block_entries'}
    assert report['block_entries'] == progress['block_entries']
    assert set(report['block_entries']) == {'capture','unlogged_1','unlogged_2','logged'}
    for row in report['block_entries'].values():
        assert [row[str(i)] for i in range(40)] == [1]*21+[0]*19
    assert report['total_block_entries'] == 84
    assert report['baseline_unchanged'] and report['old_tolerances_unchanged'] and not report['port_success']
    assert report['hardware'] == protocol['hardware']
    groups = [('pilot',root/'wan_centered_pilot_camel_s1',protocol['inputs_sha256'])]
    for group, entries in protocol['control_sha256'].items():
        name = group.removesuffix('_artifacts')
        folder = {'gradient':'gradients','retry':'prefix_camel_s1_retry1'}.get(name,name)
        folder = 'wan_centered_'+folder if name == 'retry' else f'wan_centered_{folder}_camel_s1'
        groups.append((name,root/folder,entries))
    groups.append(('source',Path(__file__).parent,
        {**protocol['helper_sources_sha256'],'wan_centered_repeatability.py':protocol['script_sha256']}))
    provenance = {}
    for group, folder, entries in groups:
        for name, expected in entries.items():
            p = folder/name
            status = ('match' if hashlib.sha256(p.read_bytes()).hexdigest()==expected else 'MISMATCH') if p.is_file() else 'not supplied locally'
            provenance[group+'/'+name] = status
    assert 'MISMATCH' not in provenance.values(), provenance
    control = root/'wan_centered_block_input_camel_s1'
    for name in ('before_flow.npz','before_target_log_probability.npz','optimization_support.npz'):
        a,b = arrays(run/name), arrays(control/name)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
    support = arrays(run/'optimization_support.npz')['common_torso']
    assert support.dtype == np.bool_ and int(support.sum()) == 324
    assert replay['before_flow_exact'] and replay['target_log_probability_exact']
    logp = arrays(run/'before_target_log_probability.npz')['forward']
    np.testing.assert_allclose(-logp[support].astype(float).mean(),replay['loss'],rtol=1e-6)
    torso = root/'wan_centered_torso_camel_s1'
    baseline = arrays(torso/'forward_latent_gradient.npz')['gradient']
    reverse = arrays(torso/'reverse_latent_gradient.npz')['gradient']
    retry = arrays(root/'wan_centered_prefix_camel_s1_retry1/forward_latent_gradient.npz')['gradient']
    gradients = {p:arrays(run/f'{p}_latent_gradient.npz')['gradient'] for p in ('unlogged_1','unlogged_2','logged')}
    refs = dict(archive=baseline,retry=retry,**gradients)
    comparisons = {}
    assert read(run/'comparisons.json') == report['comparisons']
    for phase, row in report['comparisons'].items():
        assert gradients[phase].shape == (1,16,6,60,104)
        comparisons[phase] = {}
        for name, expected in row.items():
            actual = metrics(gradients[phase],refs[name.removeprefix('vs_')])
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-12)
            comparisons[phase][name] = dict(**actual,
                angle_degrees=float(np.degrees(np.arccos(np.clip(actual['cosine'],-1,1)))))
    logged = read(run/'logged_squared_norms.json')
    old = read(root/'wan_centered_prefix_camel_s1_retry1/forward_squared_norms.json')
    energy = {}
    for name, values in logged.items():
        a,b = np.asarray(values,float),np.asarray(old[name],float)
        assert a.shape == (6,) and np.isfinite(a).all() and (a>=0).all()
        energy[name] = dict(frame0_fraction=float(a[0]/a.sum()),
            retry_frame0_fraction=float(b[0]/b.sum()),
            per_frame_energy_relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
    np.testing.assert_allclose(logged['block_20'],arrays(control/'gradient_moments.npz')['block_input'][:,0],rtol=1e-12,atol=1e-12)
    for phase, grad in gradients.items():
        v=(grad.astype(float)**2).sum(axis=(0,1,3,4))
        energy[phase+'_latent'] = dict(frame0_fraction=float(v[0]/v.sum()))
    cross = {'archived_forward_vs_archived_reverse':metrics(baseline,reverse)}
    for phase, grad in gradients.items():
        cross[phase+'_vs_archived_reverse'] = metrics(grad,reverse)
    return dict(provenance=provenance,counts=counts,total_block_entries=84,hardware=report['hardware'],
        comparisons=comparisons,temporal_energy=energy,cross_reference_context=cross,
        limits='Only the original reference was backpropagated in this run. Cross-reference context reuses '
        'the archived reverse latent gradient, not a new reverse pass. Logged block energies are not full '
        'feature gradients. No new two-reference intermediate cosines or motion outcomes are available. '
        'Unlogged passes retain constant module counters/input references. One repeat pair and one logged '
        'pass do not calibrate a statistical tolerance or isolate pass-order effects.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='probe_runs/wan_centered_repeatability_camel_s1')
    p.add_argument('--root',default='probe_runs')
    args=p.parse_args()
    result=review(args.run,args.root)
    output=Path(args.run)/'review';output.mkdir(exist_ok=True)
    (output/'repeatability_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Verified hashes:',sum(v=='match' for v in result['provenance'].values()))
    print('Missing locally:',[k for k,v in result['provenance'].items() if v!='match'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('provenance','hardware')},indent=2))


if __name__=='__main__':
    main()
