"""One offline destination-column centering experiment on saved step-39 Q/K."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.inspect_wan_static_matches import sha
from guidance_utils.wan_reference_diagnostics import comparison


def summarize(rows, criteria, variants=('native_saved', 'baseline_cpu', 'column_centered')):
    summaries, failures = [], []
    for variant in variants:
        failed = []
        for control in ('forward', 'reverse', 'static'):
            for region in ('subject', 'background', 'subject_relative_to_background'):
                subset = [r for r in rows if r['variant'] == variant and r['control'] == control
                          and r['region'] == region and not r['corrupt_endpoint']]
                row = dict(variant=variant, control=control, region=region, pairs=len(subset))
                for key in ('amf_dx', 'image_dx', 'cosine', 'epe'):
                    values = [r[key] for r in subset if r[key] is not None]
                    row[key] = float(np.mean(values)) if values else None
                summaries.append(row)
                if len(subset) != (5 if control == 'static' else 4) or any(
                        r['patches'] < criteria['min_patches_per_pair'] for r in subset):
                    failed.append(f'{control}/{region}: insufficient support')
                if row['epe'] is None or not np.isfinite(row['epe']):
                    failed.append(f'{control}/{region}: invalid error')
                if control == 'static':
                    if row['epe'] is None or row['epe'] > criteria['static_epe_max']:
                        failed.append(f'static/{region}: error exceeds one patch')
                elif region == 'subject_relative_to_background':
                    if any(r['amf_dx'] is None or r['image_dx'] is None or
                           not np.isfinite(r['amf_dx'] * r['image_dx']) or
                           r['amf_dx'] * r['image_dx'] <= 0 for r in subset):
                        failed.append(f'{control}: incorrect sign in one or more pairs')
                    if row['cosine'] is None or not np.isfinite(row['cosine']) or row['cosine'] < criteria['relative_cosine_min']:
                        failed.append(f'{control}: cosine below threshold')
                    gain = abs(row['amf_dx'] / row['image_dx']) if row['image_dx'] and row['amf_dx'] is not None else 0
                    if not criteria['relative_amplitude_range'][0] <= gain <= criteria['relative_amplitude_range'][1]:
                        failed.append(f'{control}: amplitude outside allowed range')
        failures.append(dict(variant=variant, passes_readout_screen=not failed, failures=failed))
    return summaries, failures


def inspect(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / 'plan.json').read_text())
    assert plan['grid'] == [6, 30, 52] and plan['states'] == ['step_39'] and plan['block'] == 20
    assert plan['controls'] == ['forward', 'reverse', 'static']
    assert sha(run / 'rgb_motion.npz') == plan['input_sha256']['rgb_motion.npz']
    protocol = dict(hypothesis='Source-independent destination attraction contributes to wrong peaks and diffuse tails.',
        intervention='For each frame pair, L_centered[i,j] = L[i,j] - mean_i(L[i,j]); all 1560 source positions.',
        expected='Reduce static error and many-to-one wrong matches while preserving forward/reverse direction and magnitude.',
        max_archives=3, adjacent_pairs_per_archive=5, candidate_count=1, model_calls=0, generations=0,
        temperature=2, grid=plan['grid'], block=20, heads='all 40, mean pre-softmax logits',
        checkpoint_revision=plan['checkpoint_revision'], conditioning_sha256=plan['conditioning_sha256'],
        sigma=plan['sigmas'][39], criteria=plan['criteria'],
        evaluation='Unchanged RGB-valid torso/fence support, existing corrupt endpoint exclusions; no mask enters readout.',
        limitation='Changes the AMF definition. Offline CPU BF16 replay with FP32 centering/softmax; no gradient or generation claim.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps(protocol), flush=True)
    torch.set_num_threads(8)
    frames, height, width = plan['grid']; spatial = height * width
    yy, xx = np.indices((height, width))
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    coord_tensor = torch.from_numpy(coords)
    with np.load(run / 'rgb_motion.npz') as rgb:
        regions = {r: rgb[r + '_region'].ravel() for r in ('subject', 'background')}
        truth = {c: rgb[c + '_flow'].reshape(5, spatial, 2) for c in plan['controls']}
        valid = {c: rgb[c + '_valid'].reshape(5, spatial) for c in plan['controls']}
    indices = np.flatnonzero(regions['subject'] | regions['background'])
    previous = json.loads((run / 'review/static_matches/static_matches.json').read_text())
    example_rows = []
    for region in regions:
        rr = [r for r in previous['rows'] if r['temperature'] == 2 and r['pair'] == 2 and r['region'] == region]
        for category in ('exact_peak', 'distant_peak'):
            selected = [r for r in rr if r['exact_peak']] if category == 'exact_peak' else [r for r in rr if not r['peak_within_one_patch']]
            example_rows.append(dict(max(selected, key=lambda r: r['soft_error'] if category == 'exact_peak' else r['peak_distance']), category=category))
    rows, static_rows, examples, hashes, replay = [], [], [], {}, []
    peak_agreement, centered_residuals = [], []
    for control in plan['controls']:
        archive = run / 'readouts' / f'{control}_step_39.npz'
        hashes[control] = sha(archive)
        with np.load(archive) as data:
            assert data['query'].shape == data['key'].shape == (1, 9360, 40, 128)
            q = data['query'][0].reshape(frames, spatial, -1)
            k = data['key'][0].reshape(frames, spatial, -1)
            native_soft, native_hard = data['soft'], data['hard']
        for pair in range(5):
            print(f'{control}: latent pair {pair}->{pair+1}', flush=True)
            logits = ((torch.from_numpy(q[pair]).bfloat16() @ torch.from_numpy(k[pair+1]).bfloat16().T)
                      * (1 / (40 * np.sqrt(128)))).float()
            centered = logits - logits.mean(0, keepdim=True)
            residual = float(centered.mean(0).abs().max())
            assert residual < 1e-5
            centered_residuals.append(residual)
            p0 = (logits[indices] * 2).softmax(-1)
            p1 = (centered[indices] * 2).softmax(-1)
            assert torch.isfinite(p0).all() and torch.isfinite(p1).all()
            assert torch.allclose(p1.sum(-1), torch.ones(len(indices)), atol=1e-6)
            soft0 = (p0 @ coord_tensor - coord_tensor[indices]).numpy()
            soft1 = (p1 @ coord_tensor - coord_tensor[indices]).numpy()
            native = native_soft[pair * frames + pair + 1, indices]
            hard_native = native_hard[pair * frames + pair + 1, indices]
            peak0, peak1 = p0.argmax(-1).numpy(), p1.argmax(-1).numpy()
            hard0, hard1 = coords[peak0] - coords[indices], coords[peak1] - coords[indices]
            replay.append(float(np.abs(soft0 - native).max()))
            peak_agreement.extend(np.all(hard0 == hard_native, axis=-1).tolist())
            support = valid[control][pair, indices]
            bg = regions['background'][indices] & support
            image_flow = truth[control][pair, indices]
            for variant, flow in (('native_saved', native), ('baseline_cpu', soft0), ('column_centered', soft1)):
                for region in ('subject', 'background', 'subject_relative_to_background'):
                    selection = regions['subject' if region.startswith('subject') else 'background'][indices] & support
                    a, b = flow, image_flow
                    if region == 'subject_relative_to_background':
                        assert bg.any()
                        a, b = a - np.median(a[bg], axis=0), b - np.median(b[bg], axis=0)
                    metrics = comparison(a, b, selection)
                    metrics['epe'] = float(np.linalg.norm(a[selection] - b[selection], axis=-1).mean()) if selection.any() else None
                    rows.append(dict(control=control, pair=pair, variant=variant, region=region,
                        corrupt_endpoint=(control == 'forward' and pair == 0) or (control == 'reverse' and pair == 4), **metrics))
            if control == 'static':
                for variant, hard, peaks in (('baseline_cpu', hard0, peak0), ('column_centered', hard1, peak1)):
                    for region in regions:
                        selection = regions[region][indices] & support
                        distant = selection & (np.linalg.norm(hard, axis=-1) > 1)
                        top = Counter(peaks[distant].tolist()).most_common(1)
                        static_rows.append(dict(pair=pair, variant=variant, region=region, patches=int(selection.sum()),
                            distant_peaks=int(distant.sum()), exact_peaks=int((selection & (np.linalg.norm(hard, axis=-1) == 0)).sum()),
                            hard_epe=float(np.linalg.norm(hard[selection] - image_flow[selection], axis=-1).mean()),
                            dominant_wrong_destination=coords[top[0][0]].tolist() if top else None,
                            dominant_wrong_count=top[0][1] if top else 0))
                if pair == 2:
                    for old in example_rows:
                        i = int(np.flatnonzero(indices == old['token'])[0])
                        assert np.array_equal(coords[peak0[i]], old['native_peak'])
                        native_index = int(old['native_peak'][1] * width + old['native_peak'][0])
                        assert float(p0[i, native_index]) == float(p0[i].max()), 'Original marker is not a maximum'
                        entry = dict(region=old['region'], category=old['category'], source=old['source'])
                        for tag, probability, peaks, flow in (('baseline', p0, peak0, soft0), ('centered', p1, peak1, soft1)):
                            entry[tag] = dict(peak=coords[peaks[i]].tolist(), peak_probability=float(probability[i].max()),
                                source_probability=float(probability[i, old['token']]),
                                expected_destination=(flow[i] + coords[indices[i]]).tolist())
                        examples.append(entry)
                        np.savez_compressed(output / f"example_{old['region']}_{old['category']}.npz",
                                            baseline=p0[i].numpy().reshape(height, width), centered=p1[i].numpy().reshape(height, width))
        del q, k, native_soft, native_hard
    summary, screens = summarize(rows, plan['criteria'])
    result = dict(protocol=protocol, archive_sha256=hashes, model_calls=0, port_success=False,
        baseline_replay_max_abs_error=max(replay), baseline_peak_agreement=float(np.mean(peak_agreement)),
        centered_column_mean_max_abs=max(centered_residuals), summary=summary, screens=screens,
        rows=rows, static_peak_rows=static_rows, examples=examples)
    (output / 'destination_bias.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    plot(run, output, examples, width, height)
    for row in summary:
        if row['region'] == 'subject_relative_to_background':
            print(json.dumps(row), flush=True)
    print(json.dumps(screens, indent=2), flush=True)
    print('CPU baseline max error:', max(replay), 'peak agreement:', result['baseline_peak_agreement'], flush=True)
    return result


def plot(run, output, examples, width, height):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from PIL import Image
    frame = np.asarray(Image.open(run / 'controls/static/00000.png').convert('RGB'))
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), layout='constrained')
    for row, ex in enumerate(examples):
        ax = axes[row, 0]
        ax.imshow(frame, extent=(-.5, width-.5, height-.5, -.5))
        for xy, marker, color, label in ((ex['source'], 'o', 'cyan', 'Source / correct static destination'),
                                       (ex['baseline']['peak'], 'x', 'lime', 'Original strongest match'),
                                       (ex['baseline']['expected_destination'], '+', '#ff4444', 'Original soft average')):
            ax.scatter(*xy, marker=marker, color=color, facecolors='none' if marker == 'o' else color, s=100, linewidths=2, label=label)
        ax.set_title(f"{ex['region']}: {ex['category'].replace('_', ' ')}\nSource {ex['source']}")
        with np.load(output / f"example_{ex['region']}_{ex['category']}.npz") as arrays:
            for column, tag in enumerate(('baseline', 'centered'), 1):
                im = axes[row, column].imshow(arrays[tag], norm=LogNorm(vmin=1e-6, vmax=1), cmap='magma', interpolation='nearest')
                entry = ex[tag]
                axes[row, column].set_title(f"{tag}: peak {entry['peak']}, P={entry['peak_probability']:.3f}\n"
                                           f"P(correct static destination)={entry['source_probability']:.3f}")
                # No markers cover the tiny peak cells; coordinates are in the title.
        for ax in axes[row]:
            ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), xlabel='Patch x', ylabel='Patch y')
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.colorbar(im, ax=axes[:, 1:], label='Probability per destination patch (log scale)', shrink=.7)
    fig.suptitle('Same four static examples, latent frame 2 -> 3\n'
                 'Unmarked probability maps expose the peaks; centering is an offline readout experiment')
    fig.savefig(output / 'destination_bias_examples.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.run, args.output)
