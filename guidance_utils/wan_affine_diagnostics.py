"""Known affine image motion and detached per-head AMF readout. No guidance changes."""
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

CONTROLS = ('static', 'pan_right', 'pan_left', 'rotate_cw', 'rotate_ccw', 'expand', 'contract')


def affine_controls(first, kind, count=21):
    """Render base -> frame transforms; positive rotation is clockwise in image coordinates.

    PIL uses output -> input sampling, hence the inverse passed to transform.
    No wraparound. All frames are resampled directly from the same base image.
    """
    if kind not in CONTROLS or count < 2:
        raise ValueError('Unknown control or fewer than two frames')
    first = Image.fromarray(np.asarray(first, dtype=np.uint8)).convert('RGB')
    width, height = first.size
    center = np.array([width / 2, height / 2])
    frames, matrices = [], []
    for index in range(count):
        fraction = index / (count - 1)
        angle = np.deg2rad(20 * fraction * (1 if kind == 'rotate_cw' else -1)) if kind.startswith('rotate') else 0.
        scale = 1.25 ** (fraction * (1 if kind == 'expand' else -1)) if kind in ('expand', 'contract') else 1.
        linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift = np.array([4 * index * (1 if kind == 'pan_right' else -1), 0.]) if kind.startswith('pan') else np.zeros(2)
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        matrix[:2, 2] = center - linear @ center + shift
        inverse = np.linalg.inv(matrix)
        frame = first.transform(first.size, Image.Transform.AFFINE, tuple(inverse[:2].ravel()),
                                resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
        frames.append(np.asarray(frame)); matrices.append(matrix.tolist())
    return frames, dict(kind=kind, width=width, height=height, num_frames=count,
                        base_to_frame=matrices, rotation_degrees_total=20 if kind=='rotate_cw' else -20 if kind=='rotate_ccw' else 0,
                        scale_final=1.25 if kind=='expand' else .8 if kind=='contract' else 1.,
                        coordinate_convention='pixel edge coordinates; x right, y down; clockwise positive')


def affine_truth(info, grid, temporal_stride=4, anchor_offset=0, margin_patches=2):
    """Analytic source -> target displacement at patch centers, in patch units.

    Border mask checks original content and destination visibility. VAE temporal
    anchors remain approximate; offsets test sensitivity without another model pass.
    """
    frames, h, w = grid
    if (frames-1)*temporal_stride != info['num_frames']-1:
        raise ValueError('Latent grid and decoded-frame temporal stride do not agree')
    size = np.array([info['width'], info['height']], dtype=float)
    patch = size / [w, h]
    yy, xx = np.meshgrid((np.arange(h)+.5)*patch[1], (np.arange(w)+.5)*patch[0], indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), np.ones(h*w)], axis=-1)
    matrices = np.asarray(info['base_to_frame'])
    anchors = np.clip(np.arange(frames)*temporal_stride+anchor_offset, 0, info['num_frames']-1)
    flows, masks = [], []
    def inside(p):
        return ((p[:, :2] >= margin_patches*patch) & (p[:, :2] <= size-margin_patches*patch)).all(-1)
    for source, target in zip(anchors[:-1], anchors[1:]):
        base = points @ np.linalg.inv(matrices[source]).T
        destination = base @ matrices[target].T
        flows.append((destination[:, :2]-points[:, :2])/patch)
        masks.append(inside(points) & inside(base) & inside(destination))
    return np.asarray(flows, dtype=np.float32), np.asarray(masks), anchors.tolist()


def texture_support(frames, grid, temporal_stride=4, anchor_offset=0):
    """Independent local intensity variance; do not select on predicted AMF."""
    f, h, w = grid
    masks = []
    for index in np.clip(np.arange(f-1)*temporal_stride+anchor_offset, 0, len(frames)-1):
        gray = np.asarray(Image.fromarray(frames[index]).convert('L'), dtype=np.float32)
        mean = np.asarray(Image.fromarray(gray).resize((w, h), Image.Resampling.BOX))
        square_mean = np.asarray(Image.fromarray(gray**2).resize((w, h), Image.Resampling.BOX))
        masks.append((square_mean-mean**2 >= 16).ravel())
    return np.asarray(masks)


def field_metrics(prediction, truth, selected):
    p, t = prediction[selected].astype(float), truth[selected].astype(float)
    result = dict(patches=len(p), finite_fraction=float(np.isfinite(p).all(-1).mean()) if len(p) else None)
    if not len(p) or not np.isfinite(p).all():
        return dict(result, epe=None, direction_cosine=None, positive_projection_fraction=None,
                    amplitude_ratio=None, zero_fraction=None, moving_patches=0)
    pn, tn = np.linalg.norm(p, axis=-1), np.linalg.norm(t, axis=-1)
    moving = tn >= .125
    cosines = np.zeros(len(p))  # A zero prediction is a failure on moving truth.
    nonzero = moving & (pn > 1e-8)
    cosines[nonzero] = (p[nonzero]*t[nonzero]).sum(-1)/(pn[nonzero]*tn[nonzero])
    return dict(result, epe=float(np.linalg.norm(p-t, axis=-1).mean()),
                direction_cosine=float(cosines[moving].mean()) if moving.any() else None,
                positive_projection_fraction=float(((p[moving]*t[moving]).sum(-1)>0).mean()) if moving.any() else None,
                amplitude_ratio=float((p[moving]*t[moving]).sum()/(t[moving]**2).sum()) if moving.any() else None,
                zero_fraction=float((pn<1e-8).mean()), moving_patches=int(moving.sum()))


def head_readouts(query, key, grid, temperature=2.):
    """FP32 detached measurement; head logits retain their natural 1/sqrt(D) scale.

    mean_logits is the existing AMF observer, not the mean of head probabilities.
    Running one head at a time bounds memory and leaves model tensors untouched.
    """
    from guidance_utils.motion_probe import adjacent_attention
    f, h, w = grid
    yield 'mean_logits', adjacent_attention(query, key, h, w, f, temperature)
    for head in range(query.shape[-2]):
        yield f'head_{head:02d}', adjacent_attention(query[:, :, head:head+1], key[:, :, head:head+1], h, w, f, temperature)


class AffineObserver:
    """Read-only Wan processor observer, active only during an explicitly selected pass."""
    rope_enabled = False
    context = None

    def __init__(self, root, grid, temperature, rows):
        self.root, self.grid, self.temperature, self.rows = Path(root), grid, temperature, rows
        self.active = None
        self.seen = set()

    def attention(self, block_name, query, key, injected=False):
        if self.active is None:
            return
        if injected:
            raise ValueError('Known-motion readout must not inject reference KV')
        if block_name in self.seen:
            raise ValueError(f'Duplicate readout for {block_name}')
        self.seen.add(block_name)
        labels, truths = self.active
        folder = self.root/labels['control']/labels['noise_label']/block_name
        folder.mkdir(parents=True, exist_ok=True)
        for variant, arrays in head_readouts(query, key, self.grid, self.temperature):
            np.savez_compressed(folder/f'{variant}.npz', **arrays)
            for offset, (truth, geometric, textured) in truths.items():
                for support, selected in [('geometry', geometric), ('textured', geometric & textured)]:
                    for field in ('hard', 'soft'):
                        metrics = field_metrics(arrays[field], truth, selected)
                        self.rows.append(dict(**labels, block=block_name, variant=variant, field=field,
                                              anchor_offset=offset, support=support, **metrics,
                                              entropy=float(arrays['entropy'][selected].mean()) if selected.any() else None,
                                              confidence=float(arrays['confidence'][selected].mean()) if selected.any() else None))


def make_affine_report(root):
    """Offline, self-contained HTML + CSV; all head scores retained, no winner selection."""
    root = Path(root)
    rows = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if not rows:
        raise ValueError('No completed readout rows')
    with (root/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from probe_report import _inline_figure
    backbone = html.escape(str(metadata.get('backbone', 'Wan')))
    page = [f'<!doctype html><meta charset="utf-8"><title>{backbone} affine motion readout</title>',
            '<style>body{font:16px system-ui;max-width:1500px;margin:24px auto}img{max-width:100%}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
            '<h1>Known-motion AMF readout</h1><p>Observation only: no latent optimization, KV injection, RoPE modification or target generation. '
            'The controls are image-plane transforms, not 3D turns or gaits. Gaussian noise is added to encoded controls; these are not actual sampled generation states. '
            'One noise tensor is shared across controls/noise levels; independent entries across video time. At sigma=1, all controls have identical inputs.</p>',
            '<p>EPE is endpoint error in patch units (lower better). Cosine: +1 aligned, 0 perpendicular or zero, -1 reversed. '
            'Amplitude ratio: 1 correct projected magnitude, 0 absent, negative reversed. Static controls use EPE/zero fraction, not cosine. '
            'Hard fields test argmax correspondence; soft fields test the expectation used by guidance. No nonzero-AMF filter hides static failures.</p>',
            '<p>Plots use textured, geometrically valid patches at nominal VAE anchors 0,4,...,20. CSV includes unfiltered geometry support and offsets -2/0/+2. '
            'Anchors approximate VAE temporal support; check pair fields and offset sensitivity. Head scores are descriptive, not a selected replacement for the baseline. '
            'Single-head softmax sharpness differs naturally from averaged logits. The same saved text conditions every pass; blank is the default, matching reference extraction.</p>',
            '<details><summary>Environment, source hashes and actual sigmas</summary><pre>'+html.escape(json.dumps(metadata, indent=2))+'</pre></details>']
    controls = list(dict.fromkeys(r['control'] for r in rows))
    noises = list(dict.fromkeys(r['noise_label'] for r in rows))
    if (root/'base_frame.png').is_file():
        fig, axes = plt.subplots(2, len(controls), figsize=(3*len(controls), 5), squeeze=False)
        base = np.asarray(Image.open(root/'base_frame.png').convert('RGB'))
        for col, control in enumerate(controls):
            frames, _ = affine_controls(base, control)
            for row, index in enumerate((0, 20)):
                axes[row, col].imshow(frames[index]); axes[row, col].axis('off')
                axes[row, col].set_title(f'{control} | frame {index}')
        fig.tight_layout(); page.append(_inline_figure(fig))
    # With shared noise, sigma=1 carries no control-specific information. Scores
    # may differ against different truths, but predicted fields should agree.
    for state in metadata.get('noise_states', []):
        if state['sigma'] != 1. or len(controls) < 2:
            continue
        for block in dict.fromkeys(r['block'] for r in rows):
            files = [root/c/state['noise_label']/block/'mean_logits.npz' for c in controls]
            if all(p.is_file() for p in files):
                with np.load(files[0]) as archive:
                    first_hard, first_soft = archive['hard'].copy(), archive['soft'].copy()
                agreements, errors = [], []
                for path in files[1:]:
                    with np.load(path) as archive:
                        agreements.append(float((archive['hard']==first_hard).all(-1).mean()))
                        errors.append(float(np.max(np.abs(archive['soft']-first_soft))))
                page.append(f'<p>Pure-noise sanity, {html.escape(block)}: minimum hard-field agreement across controls '
                            f'{min(agreements):.4f}; maximum soft-field difference {max(errors):.6g}. '
                            'Expected 1 and 0 for deterministic identical forwards. A high motion score here is not recovery of that control.</p>')
    for block in dict.fromkeys(r['block'] for r in rows):
        for field in ('hard', 'soft'):
            variants = list(dict.fromkeys(r['variant'] for r in rows))
            fig, axes = plt.subplots(1, len(noises), figsize=(5*len(noises), 5), squeeze=False)
            for ax, noise in zip(axes[0], noises):
                matrix = np.full((len(variants), len(controls)), np.nan)
                for r in rows:
                    if r['block']==block and r['field']==field and r['noise_label']==noise and r['support']=='textured' and r['anchor_offset']==0:
                        matrix[variants.index(r['variant']), controls.index(r['control'])] = np.nan if r['direction_cosine'] is None else r['direction_cosine']
                plot = ax.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(controls)), controls, rotation=60, ha='right')
                ax.set_yticks(range(len(variants)), variants); ax.set_title(noise)
            fig.colorbar(plot, ax=list(axes[0]), label='Direction cosine', shrink=.7)
            fig.suptitle(block+' | '+field+' | static is blank by definition')
            fig.subplots_adjust(bottom=.28, right=.85, wspace=.5)
            page.append(_inline_figure(fig))
    page.append('<h2>Current head average, nominal anchors, textured support</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in ['Control','Noise','Block','Field','Patches','EPE','Cosine','Amplitude','Zero fraction'])+'</tr>')
    def format_value(value):
        return 'n/a' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)
    for r in rows:
        if r['variant']=='mean_logits' and r['anchor_offset']==0 and r['support']=='textured':
            page.append('<tr>'+''.join('<td>'+html.escape(format_value(r[k]))+'</td>' for k in ['control','noise_label','block','field','patches','epe','direction_cosine','amplitude_ratio','zero_fraction'])+'</tr>')
    page.append('</table><p>See fields NPZ files for every adjacent pair and head; truth.npz for geometric/texture masks and known flow. '
                'Do not interpret a best head on these clips as held-out validation. If rotation/scale fail while translation passes, localize extraction before changing generation. '
                'If clean signals pass but noisy signals fail, investigate noise-level/conditioning dependence. Passing this diagnostic does not prove faithful motion transfer.</p>')
    result = root/'affine_report.html'; result.write_text('\n'.join(page), encoding='utf-8')
    return result
