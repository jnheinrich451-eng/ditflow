"""CPU gates for the bounded same-reference repeatability control."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from benchmark import wan_centered_repeatability as experiment
from verify_wan_centered_prefix import fixture


def runner_checks():
    g,ready=fixture()
    ready['retry_gradient']=ready['latent_gradients']['forward'].copy()
    native=experiment.destination_objectives
    calls=[]
    def readout(q,k,refs,ids):
        return native(q,k,refs,ids,1,4)
    def capture(model,x):
        calls.append(1)
        return model.transformer(x)
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(experiment,'prepare_repeatability_inputs',return_value=ready), \
         patch.object(experiment,'frozen_model',return_value={}), \
         patch.object(experiment,'device_details',return_value={'test_device':'cpu'}), \
         patch.object(experiment,'capture',side_effect=capture), \
         patch.object(experiment,'destination_objectives',side_effect=readout), \
         patch('benchmark.wan_centered_pilot.clear',side_effect=lambda m:m._clear_kv(range(40))):
        root=Path(tmp)
        args=(g,*[root/n for n in ('p','a','h','g','r','d','t','q','s','b','retry')])
        result=experiment.run_repeatability(*args,root/'ok')
        assert len(calls)==1
        assert result['counts']==dict(positive_forwards_attempted=1,positive_forwards_completed=1,
            latent_backwards_attempted=3,latent_backwards_completed=3)
        assert result['total_block_entries']==84 and result['old_tolerances_unchanged']
        assert set(result['block_entries'])=={'capture',*experiment.PASSES}
        for row in result['block_entries'].values():
            assert list(row.values())==[1]*21+[0]*19
        for phase in experiment.PASSES:
            value=np.load(root/'ok'/f'{phase}_latent_gradient.npz')['gradient']
            np.testing.assert_array_equal(value,ready['latent_gradients']['forward'])
        assert all(not b._forward_pre_hooks for b in g.transformer.blocks)
        assert all(p.grad is None for p in g.transformer.parameters())
        assert not result['port_success'] and result['baseline_unchanged']
        json.dumps(result,allow_nan=False)
        try:
            experiment.run_repeatability(*args,root/'ok')
        except RuntimeError as error:
            assert 'budget already' in str(error)
        else:
            raise AssertionError('Repeated budget accepted')
        assert len(calls)==1
        # An archive mismatch is measured here; the old prefix gate is NOT weakened.
        shifted={**ready,'latent_gradients':{**ready['latent_gradients'],
                  'forward':ready['latent_gradients']['forward']*1.1}}
        with patch.object(experiment,'prepare_repeatability_inputs',return_value=shifted):
            measured=experiment.run_repeatability(*args,root/'archive_drift')
        assert not measured['comparisons']['unlogged_1']['vs_archive']['old_elementwise_gate_passed']
        assert measured['comparisons']['logged']['vs_unlogged_1']['max_abs']==0
        bad={**ready,'before_logp':{a:v+1 for a,v in ready['before_logp'].items()}}
        with patch.object(experiment,'prepare_repeatability_inputs',return_value=bad):
            try:
                experiment.run_repeatability(*args,root/'bad')
            except AssertionError:
                pass
            else:
                raise AssertionError('Before-state mismatch accepted')
        failed=json.loads((root/'bad/failure.json').read_text())
        assert failed['counts']['latent_backwards_attempted']==0
        assert all(not b._forward_pre_hooks for b in g.transformer.blocks)
    print('PASS: real tiny Wan same-graph three-pass backward, exact hook/no-hook equality, 84 entries, repeat/replay/cleanup guards; archive differences measured without changing prefix thresholds.')


def preflight_checks():
    grad=np.ones((1,2,6,2,2),np.float32)
    flow=np.zeros((5,4,2),np.float32);mask=np.ones((5,4),bool)
    ready=dict(inputs=({}, {},{}, {},{}, {'before':flow}),provenance={},support=mask,
        before_logp={a:np.full((5,4),-2.,np.float32) for a in ('forward','reverse')},
        latent_gradients={'forward':grad})
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        protocol=dict(inputs_sha256={},control_sha256={},helper_sources_sha256={},
            script_sha256=experiment.digest(Path(experiment.__file__).with_name('wan_centered_prefix.py')))
        failed=dict(error_type='AssertionError',counts=dict(positive_forwards_attempted=1,positive_forwards_completed=1,
            latent_backwards_attempted=1,latent_backwards_completed=1))
        diff=experiment.difference(grad*1.01,grad)
        replay={'arms':{'forward':{k:diff[n] for k,n in [('latent_max_abs','max_abs'),('latent_rms','rms'),
            ('latent_relative_rms','relative_rms'),('latent_cosine','cosine')]}}}
        for n,v in [('started',protocol),('failure',failed),('replay',replay),('forward_squared_norms',{})]:
            experiment.write_json(root/(n+'.json'),v)
        np.savez_compressed(root/'before_flow.npz',flow=flow)
        np.savez_compressed(root/'optimization_support.npz',common_torso=mask)
        np.savez_compressed(root/'before_target_log_probability.npz',**ready['before_logp'])
        np.savez_compressed(root/'forward_latent_gradient.npz',gradient=grad*1.01)
        with patch.object(experiment,'prepare_prefix_inputs',side_effect=lambda *a:dict(ready)):
            checked=experiment.prepare_repeatability_inputs(*['unused']*10,root)
            assert len(checked['provenance']['retry_artifacts'])==8
            np.savez_compressed(root/'forward_latent_gradient.npz',gradient=grad*2)
            try:
                experiment.prepare_repeatability_inputs(*['unused']*10,root)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupt retry gradient accepted')
    print('PASS: retry provenance link and saved-gradient corruption rejection.')


def notebook_checks():
    import nbformat
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'wan_centered_repeatability.ipynb',as_version=4)
    nbformat.validate(nb)
    sources=[]
    for c in nb.cells:
        if c.cell_type=='code':
            assert not c.outputs and c.execution_count is None
            compile(c.source,'<cell>','exec');sources.append(c.source)
    assert any('BLOCK_INPUT, RETRY, OUTPUT)' in s for s in sources)
    assert any('total_memory' in s and 'load_pilot' in s for s in sources)
    print('PASS: clean notebook syntax/schema, retry wiring and memory gate before loading.')


if __name__=='__main__':
    torch.set_num_threads(1)
    preflight_checks()
    runner_checks()
    notebook_checks()
