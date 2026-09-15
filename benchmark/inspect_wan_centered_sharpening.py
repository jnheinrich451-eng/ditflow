"""One fixed sharpening candidate with destination centering held unchanged."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.inspect_wan_destination_bias import summarize
from benchmark.inspect_wan_static_matches import sha
from guidance_utils.wan_reference_diagnostics import comparison


def inspect(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / 'plan.json').read_text())
    prior_path = run / 'review/destination_bias/destination_bias.json'
    prior = json.loads(prior_path.read_text())
    assert plan['grid'] == [6, 30, 52] and plan['states'] == ['step_39'] and plan['block'] == 20
    assert plan['controls'] == ['forward', 'reverse', 'static']
    assert sha(run / 'rgb_motion.npz') == plan['input_sha256']['rgb_motion.npz']
    protocol = dict(
        hypothesis='Remaining diffuse probability after destination centering causes weak reverse motion and false static motion.',
        expected='Sharper readout preserves moving signs/amplitudes and reduces static error below one patch.',
        changed_factor='Logit multiplier (called temperature in this code): 2 -> 8. Larger means sharper.',
        fixed='Destination-column centering over all 1560 source positions, all heads, block 20, native saved Q/K, '
              'step39 noise, checkpoint, conditioning, ROI validity, endpoint exclusions.',
        max_archives=3, adjacent_pairs_per_archive=5, candidate_count=1, model_calls=0, generations=0,
        reference_report_sha256=sha(prior_path), criteria=plan['criteria'],
        checkpoint_revision=plan['checkpoint_revision'], conditioning_sha256=plan['conditioning_sha256'],
        sigma=plan['sigmas'][39], grid=plan['grid'],
        hard_readout='Compute existing centered argmax as a diagnostic of wrong peaks; not a differentiable candidate.',
        limitation='Offline readout of noised encoded controls, not guided sampling; no gradient or decoded acceptance claim.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps(protocol), flush=True)
    torch.set_num_threads(8)
    f, h, w = plan['grid']; hw = h * w
    yy, xx = np.indices((h, w))
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    with np.load(run / 'rgb_motion.npz') as rgb:
        masks = {r: rgb[r + '_region'].ravel() for r in ('subject', 'background')}
        truth = {c: rgb[c + '_flow'].reshape(5, hw, 2) for c in plan['controls']}
        valid = {c: rgb[c + '_valid'].reshape(5, hw) for c in plan['controls']}
    indices = np.flatnonzero(masks['subject'] | masks['background'])
    coord_tensor = torch.from_numpy(coords)
    variants = ('centered_t2', 'centered_t8', 'centered_hard_diagnostic')
    rows, replay_errors, fields, hashes = [], [], {}, {}
    for control in plan['controls']:
        archive = run / 'readouts' / f'{control}_step_39.npz'
        hashes[control] = sha(archive)
        assert hashes[control] == prior['archive_sha256'][control]
        with np.load(archive) as capture:
            assert capture['query'].shape == capture['key'].shape == (1, 9360, 40, 128)
            q = capture['query'][0].reshape(f, hw, -1)
            k = capture['key'][0].reshape(f, hw, -1)
        per_variant = {v: [] for v in variants}
        for pair in range(5):
            print(f'{control}: pair {pair}->{pair+1}', flush=True)
            logits = ((torch.from_numpy(q[pair]).bfloat16() @ torch.from_numpy(k[pair+1]).bfloat16().T)
                      * (1 / (40 * np.sqrt(128)))).float()
            centered = (logits - logits.mean(0, keepdim=True))[indices]
            flows = {}
            for temperature in (2, 8):
                probs = (centered * temperature).softmax(-1)
                assert torch.isfinite(probs).all()
                assert torch.allclose(probs.sum(-1), torch.ones(len(indices)), atol=1e-6)
                flows[f'centered_t{temperature}'] = (probs @ coord_tensor - coord_tensor[indices]).numpy()
            peaks = centered.argmax(-1).numpy()
            flows['centered_hard_diagnostic'] = coords[peaks] - coords[indices]
            support = valid[control][pair, indices]
            bg = masks['background'][indices] & support
            assert bg.any()
            image_flow = truth[control][pair, indices]
            for variant, flow in flows.items():
                per_variant[variant].append(flow)
                for region in ('subject', 'background', 'subject_relative_to_background'):
                    selected = masks['subject' if region.startswith('subject') else 'background'][indices] & support
                    a, b = flow, image_flow
                    if region == 'subject_relative_to_background':
                        a, b = a - np.median(a[bg], axis=0), b - np.median(b[bg], axis=0)
                    metrics = comparison(a, b, selected)
                    metrics['epe'] = float(np.linalg.norm(a[selected] - b[selected], axis=-1).mean()) if selected.any() else None
                    if variant == 'centered_t2':
                        old = next(r for r in prior['rows'] if r['control'] == control and r['pair'] == pair
                                   and r['region'] == region and r['variant'] == 'column_centered')
                        for key in ('amf_dx', 'image_dx', 'cosine', 'epe'):
                            if metrics[key] is None or old[key] is None:
                                assert metrics[key] is old[key]
                            else:
                                error = abs(metrics[key] - old[key])
                                assert error < 1e-6, (control, pair, region, key, error)
                                replay_errors.append(error)
                    rows.append(dict(control=control, pair=pair, variant=variant, region=region,
                        corrupt_endpoint=(control == 'forward' and pair == 0) or (control == 'reverse' and pair == 4), **metrics))
        fields.update({control + '_' + v: np.stack(a) for v, a in per_variant.items()})
        del q, k
    summary, screens = summarize(rows, plan['criteria'], variants=variants)
    result = dict(protocol=protocol, summary=summary, screens=screens, rows=rows,
        archive_sha256=hashes, centered_t2_replay_max_metric_error=max(replay_errors),
        model_calls=0, port_success=False,
        passes_candidate_readout_screen=next(s['passes_readout_screen'] for s in screens if s['variant'] == 'centered_t8'))
    (output / 'centered_sharpening.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'roi_flows.npz', indices=indices, **fields)
    plot(output, summary, variants)
    for row in summary:
        if row['region'] == 'subject_relative_to_background' or row['control'] == 'static':
            print(json.dumps(row), flush=True)
    print(json.dumps(screens, indent=2), flush=True)
    print('Previous centered-T2 metric replay max error:', max(replay_errors), flush=True)
    return result


def plot(output, summary, variants):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    colors = ('#ce7730', '#297db4', '#6c6c6c')
    labels = ('Centered, multiplier 2', 'Centered, multiplier 8', 'Centered argmax (diagnostic only)')
    for i, (variant, color, label) in enumerate(zip(variants, colors, labels)):
        values = [next(r['amf_dx'] for r in summary if r['variant'] == variant and r['control'] == c
                       and r['region'] == 'subject_relative_to_background') for c in ('forward', 'reverse', 'static')]
        axes[0].bar(np.arange(3) + (i-1)*.23, values, .22, color=color, label=label)
        errors = [next(r['epe'] for r in summary if r['variant'] == variant and r['control'] == 'static'
                       and r['region'] == region) for region in ('subject', 'background', 'subject_relative_to_background')]
        axes[1].bar(np.arange(3) + (i-1)*.23, errors, .22, color=color)
    rgb = [next(r['image_dx'] for r in summary if r['control'] == c and r['region'] == 'subject_relative_to_background')
           for c in ('forward', 'reverse', 'static')]
    axes[0].scatter(np.arange(3), rgb, marker='_', s=600, color='black', linewidths=3, zorder=5, label='RGB motion estimate')
    axes[0].axhline(0, color='black', linewidth=.7)
    axes[0].set_xticks(range(3), ['Original vanilla\n(leftward)', 'Reversed vanilla\n(rightward)', 'Static'])
    axes[0].set(ylabel='Relative horizontal displacement (patches)', title='Direction and amount must match the image')
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color='black', linestyle='--', label='Screen limit: 1 patch')
    axes[1].set_xticks(range(3), ['Torso', 'Fence', 'Torso relative\nto fence'])
    axes[1].set(ylabel='Static endpoint error (patches)', title='Static error must pass in every region')
    axes[1].legend(fontsize=8)
    fig.suptitle('Destination centering fixed; one sharper readout tested | cached step-39 Q/K, no generations')
    fig.savefig(output / 'centered_sharpening.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.run, args.output)
