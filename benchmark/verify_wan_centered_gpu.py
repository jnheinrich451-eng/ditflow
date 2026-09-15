"""Bounded CUDA check of saved centered AMF: 15 pairs and one Q/K backward."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from benchmark.inspect_wan_destination_bias import summarize
from benchmark.inspect_wan_static_matches import sha
from guidance_utils.wan_centered_amf import centered_pair_flow
from guidance_utils.wan_reference_diagnostics import comparison


def verify(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory to preserve the GPU run budget')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    plan = json.loads((run / 'plan.json').read_text())
    prior_dir = run / 'review/centered_sharpening'
    prior = json.loads((prior_dir / 'centered_sharpening.json').read_text())
    assert prior['passes_candidate_readout_screen']
    assert plan['grid'] == [6, 30, 52] and plan['states'] == ['step_39'] and plan['block'] == 20
    assert sha(run / 'rgb_motion.npz') == plan['input_sha256']['rgb_motion.npz']
    output.mkdir(parents=True)
    protocol = dict(hypothesis='Centered-T8 motion screening survives CUDA and has finite nonzero Q/K derivatives.',
        max_archives=3, max_pairs=15, max_backward=1, model_calls=0, generations=0,
        fixed=prior['protocol'], criterion='Same motion screen; finite nonzero query and key gradients; centering gradient reaches unselected queries.',
        backward_objective='Static pair 2->3: mean squared flow at RGB-valid torso/fence tokens, zero target. No latent optimization.',
        cpu_report_sha256=sha(prior_dir / 'centered_sharpening.json'),
        kernel_sha256=sha(Path(__file__).resolve().parents[1] / 'guidance_utils/wan_centered_amf.py'),
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                     gpu=torch.cuda.get_device_name(), memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, qk_dtype='bfloat16', softmax_dtype='float32'))
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps(protocol), flush=True)
    torch.cuda.reset_peak_memory_stats()
    f, h, w = plan['grid']; hw = h * w
    with np.load(run / 'rgb_motion.npz') as rgb:
        masks = {r: rgb[r + '_region'].ravel() for r in ('subject', 'background')}
        truth = {c: rgb[c + '_flow'].reshape(5, hw, 2) for c in plan['controls']}
        valid = {c: rgb[c + '_valid'].reshape(5, hw) for c in plan['controls']}
    indices = np.flatnonzero(masks['subject'] | masks['background'])
    with np.load(prior_dir / 'roi_flows.npz') as z:
        assert np.array_equal(indices, z['indices'])
        cpu = {c: z[c + '_centered_t8'] for c in plan['controls']}
    rows, differences, gradients, saved = [], [], None, {}
    for control in plan['controls']:
        archive = run / 'readouts' / f'{control}_step_39.npz'
        assert sha(archive) == prior['archive_sha256'][control]
        with np.load(archive) as z:
            assert z['query'].shape == z['key'].shape == (1, 9360, 40, 128)
            q = z['query'][0].reshape(f, hw, 40, 128)
            k = z['key'][0].reshape(f, hw, 40, 128)
        fields = []
        for pair in range(5):
            backward = control == 'static' and pair == 2
            print(f'{control} {pair}->{pair+1}, backward={backward}', flush=True)
            query = torch.tensor(q[pair], device='cuda', dtype=torch.bfloat16, requires_grad=backward)
            key = torch.tensor(k[pair+1], device='cuda', dtype=torch.bfloat16, requires_grad=backward)
            with torch.set_grad_enabled(backward):
                flow = centered_pair_flow(query, key, h, w)
                assert torch.isfinite(flow).all()
                if backward:
                    selection = (masks['subject'] | masks['background']) & valid[control][pair]
                    selection = torch.from_numpy(selection).to('cuda')
                    loss = flow[selection].square().mean()
                    loss.backward()
                    def stats(t):
                        if t is None:
                            return dict(finite=False, norm=0., nonzero_fraction=0.)
                        return dict(finite=bool(torch.isfinite(t).all()), norm=float(t.float().norm()),
                                    nonzero_fraction=float((t != 0).float().mean()))
                    gradients = dict(loss=float(loss.detach()), query=stats(query.grad), key=stats(key.grad),
                        unselected_query=stats(query.grad[~selection]) if query.grad is not None else stats(None),
                        meaning='Feature-space derivatives only; no model Jacobian or latent update tested.')
                    gradients['passed'] = all(gradients[x]['finite'] and gradients[x]['norm'] > 0
                                              for x in ('query', 'key', 'unselected_query'))
            gpu = flow.detach().float().cpu().numpy()[indices]
            fields.append(gpu)
            delta = gpu - cpu[control][pair]
            differences.append(dict(control=control, pair=pair, max_abs=float(np.abs(delta).max()),
                                    rms=float(np.sqrt(np.mean(delta ** 2)))))
            support = valid[control][pair, indices]
            bg = masks['background'][indices] & support
            image = truth[control][pair, indices]
            for region in ('subject', 'background', 'subject_relative_to_background'):
                selected = masks['subject' if region.startswith('subject') else 'background'][indices] & support
                a, b = gpu, image
                if region == 'subject_relative_to_background':
                    assert bg.any()
                    a, b = a - np.median(a[bg], axis=0), b - np.median(b[bg], axis=0)
                metrics = comparison(a, b, selected)
                metrics['epe'] = float(np.linalg.norm(a[selected] - b[selected], axis=-1).mean()) if selected.any() else None
                rows.append(dict(control=control, pair=pair, variant='centered_t8_cuda', region=region,
                    corrupt_endpoint=(control == 'forward' and pair == 0) or (control == 'reverse' and pair == 4), **metrics))
            del query, key, flow
        saved[control] = np.stack(fields)
        del q, k
    summary, screens = summarize(rows, plan['criteria'], variants=('centered_t8_cuda',))
    passed = screens[0]['passes_readout_screen'] and gradients['passed']
    report = dict(stage_a_passed=passed, port_success=False, protocol=protocol, summary=summary, screens=screens,
        gradients=gradients, cpu_gpu_differences=differences, rows=rows,
        max_flow_difference=max(d['max_abs'] for d in differences),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), model_calls=0, backward_calls=1,
        kernel_sha256=protocol['kernel_sha256'], archive_sha256=prior['archive_sha256'])
    (output / 'gpu_readout.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'roi_flows.npz', indices=indices, **saved)
    print(json.dumps(dict(stage_a_passed=passed, screens=screens, gradients=gradients,
                         max_flow_difference=report['max_flow_difference'],
                         peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes']), indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.output)
    if not result['stage_a_passed']:
        raise SystemExit('GPU readout or derivative gate failed; do not proceed to full-model pilot')
