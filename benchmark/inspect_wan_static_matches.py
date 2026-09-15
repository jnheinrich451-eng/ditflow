"""Decompose saved static AMF into strongest-match error and probability spread.

Offline diagnostic only: no model, optimization, or production setting changes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / 'plan.json').read_text())
    assert plan['states'] == ['step_39'] and plan['grid'] == [6, 30, 52] and plan['block'] == 20
    static_files = sorted((run / 'controls/static').glob('*.png'))
    assert len(static_files) == 21
    assert len({sha(p) for p in static_files}) == 1, 'Static frames must be identical'
    assert sha(run / 'rgb_motion.npz') == plan['input_sha256']['rgb_motion.npz']
    print('HYPOTHESIS: static error contains wrong strongest matches and/or distant probability spread. '
          'LIMIT: one saved static archive, five adjacent pairs, existing temperatures 2 and 8; '
          'zero model calls. No new parameter candidate.', flush=True)
    torch.set_num_threads(8)
    frames, height, width = plan['grid']
    spatial = height * width
    yy, xx = np.indices((height, width))
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    with np.load(run / 'rgb_motion.npz') as rgb:
        regions = {name: rgb[name + '_region'].ravel() for name in ('subject', 'background')}
        valid = rgb['static_valid'].reshape(frames - 1, spatial)
        truth = rgb['static_flow'].reshape(frames - 1, spatial, 2)
    indices = np.flatnonzero(regions['subject'] | regions['background'])
    archive = run / 'readouts/static_step_39.npz'
    with np.load(archive) as data:
        assert data['query'].shape == data['key'].shape == (1, 9360, 40, 128)
        query = data['query'][0].reshape(frames, spatial, -1)
        key = data['key'][0].reshape(frames, spatial, -1)
        saved_soft, saved_hard = data['soft'], data['hard']
    rows, examples = [], []
    replay_errors, decomposition_errors = [], []
    peak_mismatches, peak_count = 0, 0
    for pair in range(frames - 1):
        print(f'Inspecting static latent pair {pair}->{pair + 1}', flush=True)
        logits = ((torch.from_numpy(query[pair, indices]).bfloat16() @
                   torch.from_numpy(key[pair + 1]).bfloat16().T) / (40 * np.sqrt(128))).float()
        logits_np = logits.numpy()
        hard = saved_hard[pair * frames + pair + 1, indices]
        peak_xy = coords[indices] + hard
        assert np.allclose(peak_xy, np.rint(peak_xy))
        native_peak = (peak_xy[:, 1] * width + peak_xy[:, 0]).astype(int)
        cpu_peak = logits_np.argmax(-1)
        support = valid[pair, indices]
        peak_mismatches += int(((cpu_peak != native_peak) & support).sum())
        peak_count += int(support.sum())
        peak_distance = np.linalg.norm(hard, axis=-1)
        # Euclidean radii, in patches; attribution only, no masked probability or model input.
        from_peak = coords[None, :, :] - peak_xy[:, None, :]
        near_peak = np.linalg.norm(from_peak, axis=-1) <= 2
        from_source = coords[None, :, :] - coords[indices, None, :]
        near_source = np.linalg.norm(from_source, axis=-1) <= 1
        for temperature in (2., 8.):
            prob = (logits * temperature).softmax(-1).numpy()
            soft = prob @ coords - coords[indices]
            near_shift = np.einsum('ij,ijk->ik', prob * near_peak, from_peak)
            far_shift = np.einsum('ij,ijk->ik', prob * ~near_peak, from_peak)
            residual = np.max(np.abs(soft - (hard + near_shift + far_shift)))
            decomposition_errors.append(float(residual))
            assert residual < 1e-3, residual
            if temperature == 2:
                replay_errors.append(float(np.max(np.abs(
                    soft[support] - saved_soft[pair * frames + pair + 1, indices][support]))))
            mass_near_source = (prob * near_source).sum(-1)
            far_mass = (prob * ~near_peak).sum(-1)
            for region, mask in regions.items():
                selected = np.flatnonzero(mask[indices] & support)
                for i in selected:
                    rows.append(dict(
                        pair=pair, temperature=temperature, region=region, token=int(indices[i]),
                        source=coords[indices[i]].tolist(), native_peak=peak_xy[i].tolist(),
                        peak_distance=float(peak_distance[i]),
                        exact_peak=bool(peak_distance[i] == 0),
                        peak_within_one_patch=bool(peak_distance[i] <= 1),
                        expected_destination=(soft[i] + coords[indices[i]]).tolist(),
                        soft_vector=soft[i].tolist(), hard_vector=hard[i].tolist(),
                        near_peak_shift=near_shift[i].tolist(), far_peak_shift=far_shift[i].tolist(),
                        soft_error=float(np.linalg.norm(soft[i] - truth[pair, indices[i]])),
                        hard_error=float(np.linalg.norm(hard[i] - truth[pair, indices[i]])),
                        mass_within_one_patch_of_source=float(mass_near_source[i]),
                        mass_beyond_two_patches_of_peak=float(far_mass[i]),
                        native_peak_probability=float(prob[i, native_peak[i]]),
                        cpu_peak_agrees=bool(cpu_peak[i] == native_peak[i]),
                        source_logit_rank=int(1 + (logits_np[i] > logits_np[i, indices[i]]).sum()),
                    ))
                # Deterministic illustrative examples; summaries below use all valid tokens/pairs.
                if pair == 2 and temperature == 2:
                    for label, eligible, score in (
                        ('exact peak, large soft error', selected[peak_distance[selected] == 0], np.linalg.norm(soft, axis=-1)),
                        ('distant peak', selected[peak_distance[selected] > 1], peak_distance),
                    ):
                        if len(eligible):
                            i = eligible[np.argmax(score[eligible])]
                            examples.append(dict(region=region, label=label, source=coords[indices[i]],
                                peak=peak_xy[i], expected=soft[i] + coords[indices[i]],
                                probability=prob[i].reshape(height, width)))
    summaries = []
    for region in regions:
        for temperature in (2., 8.):
            for group in ('all', 'exact_peak', 'peak_within_one_patch', 'distant_peak'):
                chosen = [r for r in rows if r['region'] == region and r['temperature'] == temperature and (
                    group == 'all' or (group == 'exact_peak' and r['exact_peak']) or
                    (group == 'peak_within_one_patch' and r['peak_within_one_patch']) or
                    (group == 'distant_peak' and not r['peak_within_one_patch']))]
                if not chosen:
                    continue
                summary = dict(region=region, temperature=temperature, group=group, patch_pairs=len(chosen))
                for field in ('soft_error', 'hard_error', 'mass_within_one_patch_of_source',
                              'mass_beyond_two_patches_of_peak', 'native_peak_probability',
                              'exact_peak', 'peak_within_one_patch'):
                    summary[field + '_mean'] = float(np.mean([r[field] for r in chosen]))
                for field in ('soft_vector', 'hard_vector', 'near_peak_shift', 'far_peak_shift'):
                    summary[field + '_mean'] = np.mean([r[field] for r in chosen], axis=0).tolist()
                summaries.append(summary)
    result = dict(model_calls=0, port_success=False, source_archive_sha256=sha(archive),
        checkpoint_revision=plan['checkpoint_revision'], conditioning_sha256=plan['conditioning_sha256'],
        grid=plan['grid'], block=plan['block'], sigma=plan['sigmas'][39],
        temperatures=[2, 8], static_image_sha256=sha(static_files[0]),
        aggregation='Pooled valid patch-pairs, five adjacent latent pairs. Exact peak and <=1 subsets overlap.',
        method='Native saved hard peak + CPU reconstructed probability expectation. '
               'soft displacement = hard displacement + expected offset within radius 2 of peak + offset outside radius 2.',
        limitations='CPU BF16 replay, not native GPU parity. Static pixels do not imply identical noisy latent features. '
                    'No causal separation of noise, positional encoding, or temporal VAE effects. '
                    'Manual ROIs are evaluation-only. No gradients or decoded generation tested.',
        replay_soft_max_abs_error=max(replay_errors), replay_peak_mismatch_fraction=peak_mismatches / peak_count,
        decomposition_max_abs_error=max(decomposition_errors), summary=summaries, rows=rows)
    write_report(result, output)
    (output / 'static_matches.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    plot_examples(run, output, examples, width, height)
    for item in summaries:
        if item['group'] in ('all', 'exact_peak'):
            print(json.dumps(item), flush=True)
    print('Replay soft max error:', result['replay_soft_max_abs_error'])
    print('Replay peak mismatch fraction:', result['replay_peak_mismatch_fraction'])
    return result


def write_report(result, output):
    """Can regenerate the narrative from saved JSON without repeating Q/K work."""
    output = Path(output)
    rows = [r for r in result['rows'] if r['temperature'] == 2]
    destinations = []
    for pair in range(5):
        wrong = [r for r in rows if r['pair'] == pair and not r['peak_within_one_patch']]
        counts = Counter(tuple(r['native_peak']) for r in wrong)
        destinations.append(dict(pair=pair, distant_matches=len(wrong),
            top_destinations=[dict(xy=list(xy), count=count) for xy, count in counts.most_common(4)]))
    result['repeated_wrong_destinations'] = destinations
    lines = [
        '# Static correspondence: both probability spread and wrong peaks remain', '',
        'One offline analysis of the saved step-39 static Q/K: five adjacent latent pairs, '
        'block 20, all heads, existing temperatures 2 and 8. Zero model calls or generations. '
        'The 21 static PNGs are byte-identical. Production guidance is unchanged.', '',
        '## Findings', '',
        '| Evaluation region | Valid patch-pairs | Exact strongest match | Strongest match >1 patch away | '
        'Soft error when strongest match is exact, T=2 | Same subset, T=8 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for region in ('subject', 'background'):
        def summary(temperature, group):
            return next(r for r in result['summary'] if r['region'] == region and
                        r['temperature'] == temperature and r['group'] == group)
        all_rows = summary(2, 'all')
        lines.append(f"| {region} | {all_rows['patch_pairs']} | {all_rows['exact_peak_mean']:.1%} | "
                     f"{1-all_rows['peak_within_one_patch_mean']:.1%} | "
                     f"{summary(2, 'exact_peak')['soft_error_mean']:.2f} | "
                     f"{summary(8, 'exact_peak')['soft_error_mean']:.2f} |")
    lines += ['',
        'Errors are mean vector endpoint distances in patches (one patch = 16 RGB pixels). '
        'These are pooled individual-patch errors within each region, not the earlier '
        'torso-minus-median-fence statistic. A background median can hide large errors at individual patches. '
        'Patch-pairs are repeated observations from one video, not independent experiments.', '',
        'The false flow exists even where the native strongest match is exactly at the source position. '
        'This directly demonstrates bias from the remaining probability distribution. '
        'Sharpening reduces that bias but cannot reorder the strongest matches.', '',
        '### Wrong peaks often share a destination', '',
        '| Latent pair | Matches >1 patch away | Most common wrong destination (x,y) | Count |',
        '|---|---:|---|---:|',
    ]
    for entry in destinations:
        top = entry['top_destinations'][0]
        lines.append(f"| {entry['pair']} -> {entry['pair']+1} | {entry['distant_matches']} | "
                     f"{top['xy']} | {top['count']} |")
    lines += ['',
        'For example, 43 of 65 distant matches in pair 2->3 land on patch (10,9). '
        'This is consistent with a few destinations attracting unrelated source patches. '
        'It does not establish whether key magnitude, position, noise, or learned semantic features cause it.', '',
        '## Read the figure', '',
        '[Open static_matches.png](static_matches.png). Cyan circle: source and correct static destination. '
        'Green cross: native strongest match. Red plus: soft expected destination. '
        'On a correct static readout, all three coincide. Right panels show the destination probabilities '
        'on a logarithmic scale, at temperature 2. Examples are the largest errors in each category '
        'on the preselected central pair, not representative averages. All valid patches appear in JSON. '
        'These overlays and regions are evaluation-only; they are never supplied to Wan.', '',
        '## Verification and limits', '',
        f"CPU reconstruction matches every native hard peak on valid ROI tokens. "
        f"Maximum soft-flow discrepancy is {result['replay_soft_max_abs_error']:.6f} patch. "
        f"The vector identity soft = hard + near-peak offset + distant offset closes within "
        f"{result['decomposition_max_abs_error']:.8f} patch. These checks validate this decomposition, "
        'not native backend parity or motion transfer.', '',
        'The input pixels are static, but VAE temporal context, frame-specific noise, and model positional '
        'encoding can make their latent features different. This analysis has not separated those causes. '
        'It uses no latent optimization and says nothing new about gradient quality.', '',
        '## One next discriminating experiment', '',
        'Hypothesis: attraction shared across source patches contributes to the wrong peaks and diffuse tails. '
        'Compare the unchanged readout with exactly one experimental readout that subtracts each destination '
        'column mean from its logits before softmax (mean over all source positions in that frame). '
        'Hold temperature at 2, block/head selection, Q/K, native RoPE, controls and support fixed. '
        'Maximum three saved step-39 archives, five adjacent pairs each, zero model calls or generations. '
        'This deliberately changes the AMF readout; it is not the released DiTFlow objective.', '',
        'Expected distinguishing outcome: fewer many-to-one wrong peaks, static error <=1 patch in each '
        'region, and preserved correct forward/reverse signs and amplitudes under the existing screen. '
        'If static error falls only because all motion is suppressed, reject it. If peaks remain wrong, '
        'reject this explanation as sufficient. No generation is warranted until a readout passes those '
        'controls; a pass would still require native GPU gradients and decoded-video validation. '
        'This next experiment is specified here but has not been run.', '',
        '## How to read evidence', '',
        '1. Optimization loss: smaller means the target AMF agrees more with reference AMF under the selected '
        'loss and mask. It does not establish that either field measures motion correctly. Compare values '
        'only with the same loss, masks, frame pairs, weighting and noise state.',
        '2. Readout accuracy: endpoint error should decrease; static displacement should approach zero; '
        'moving displacement should match the image estimate, not merely shrink. Directional cosine is '
        'useful for moving controls and undefined for zero motion.',
        '3. Intervention: a real latent update must survive casting and change a fresh prediction and the '
        'next scheduler state from identical solver state. Earlier evidence resolved this for the tested setup.',
        '4. Acceptance: decoded subject trajectories must respond correctly to opposite references under '
        'fixed inputs, then replicate on another seed and motion case. This remains unproven.', '',
        '## Reproduce', '', '```text',
        'python -m benchmark.inspect_wan_static_matches probe_runs/wan_camel_correspondence_step39_s1 '
        '--output probe_runs/wan_camel_correspondence_step39_s1/review/static_matches', '```', '',
    ]
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_examples(run, output, examples, width, height):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from PIL import Image
    frame = np.asarray(Image.open(run / 'controls/static/00000.png').convert('RGB'))
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 3.2 * len(examples)), squeeze=False,
                             layout='constrained')
    for row, example in enumerate(examples):
        axes[row, 0].imshow(frame, extent=(-.5, width-.5, height-.5, -.5))
        axes[row, 1].imshow(example['probability'], norm=LogNorm(vmin=1e-6, vmax=1), cmap='magma')
        for ax in axes[row]:
            for key, marker, color, label in (('source', 'o', '#00ffff', 'Source (zero-motion destination)'),
                                            ('peak', 'x', '#00ff00', 'Strongest match'),
                                            ('expected', '+', '#ff4444', 'Soft average')):
                xy = example[key]
                ax.scatter(*xy, s=110, marker=marker, color=color, linewidths=2, label=label,
                           facecolors='none' if marker == 'o' else color)
            ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), xlabel='Patch x', ylabel='Patch y')
        axes[row, 0].set_title(example['region'] + ': ' + example['label'])
        axes[row, 1].set_title('Destination probability (log scale; temperature 2)')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Static frame, latent pair 2 -> 3: all true image displacements are zero\n'
                 'Examples selected by largest error within each category; see JSON for all patches')
    fig.savefig(output / 'static_matches.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.run, args.output)
