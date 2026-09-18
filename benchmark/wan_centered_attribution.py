"""Read the failed pilot's saved states: three forwards, no updates or sampling."""
import hashlib
import json
from pathlib import Path

import numpy as np


def native_frame_mass(query, key, query_indices, frames, spatial, chunk_size=8):
    """Per-query/head mass over frames using native SDPA logits and ALL valid keys.

    Q/K are native normalized post-RoPE tensors (S,H,D). FP32 reconstruction of
    the native formula, not a claim of bitwise fused-SDPA attention weights.
    This fixed T2V diagnostic supports full, unmasked video self-attention only.
    """
    import torch
    import math
    if query.ndim != 3 or query.shape != key.shape or key.shape[0] != frames*spatial:
        raise ValueError('Expected all valid video tokens without padding or text')
    selected = torch.as_tensor(query_indices, dtype=torch.long, device=query.device)
    if selected.ndim != 1 or selected.numel() == 0 or bool(((selected < 0) | (selected >= len(query))).any()):
        raise ValueError('Invalid query indices')
    result = []
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        keys = key.float().permute(1, 2, 0)
        for indices in selected.split(chunk_size):
            logits = (query[indices].float().permute(1, 0, 2) @ keys) / math.sqrt(query.shape[-1])
            denominator = logits.logsumexp(dim=-1, keepdim=True)
            numerator = logits.reshape(query.shape[1], len(indices), frames, spatial).logsumexp(dim=-1)
            result.append((numerator-denominator).exp().permute(1, 0, 2).cpu())
    return torch.cat(result).numpy()


def mass_queries():
    """Four preselected torso/fence queries in each of five source frames."""
    patches = {'torso': [(10, 24), (10, 32), (14, 24), (14, 32)],
               'fence': [(14, 2), (14, 6), (18, 2), (18, 6)]}
    return [dict(region=region, source_frame=f, destination_frame=f+1, row=y, column=x,
                 token_index=f*1560+y*52+x)
            for f in range(5) for region, coords in patches.items() for y, x in coords]


def summarize_mass(mass, queries):
    result = []
    for frame in range(5):
        for region in ('torso', 'fence'):
            ids = [i for i, q in enumerate(queries) if q['source_frame'] == frame and q['region'] == region]
            values = mass[ids, :, frame+1]
            result.append(dict(source_frame=frame, destination_frame=frame+1, region=region,
                mean=float(values.mean()), quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
                per_head_mean=values.mean(axis=0).tolist(),
                mean_mass_to_each_frame=mass[ids].mean(axis=(0, 1)).tolist()))
    return result


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def regions(height=30, width=52, reference_rgb=False):
    """Fixed conservative RGB ROIs from the reviewed camel; not a segmentation mask."""
    yy, xx = np.meshgrid((np.arange(height)+.5)*480/height,
                         (np.arange(width)+.5)*832/width, indexing='ij')
    tb, fb = ((360, 160, 500, 260), (20, 120, 180, 190)) if reference_rgb else ((350, 140, 590, 260), (0, 200, 140, 350))
    torso = (xx >= tb[0]) & (xx < tb[2]) & (yy >= tb[1]) & (yy < tb[3])
    fence = (xx >= fb[0]) & (xx < fb[2]) & (yy >= fb[1]) & (yy < fb[3])
    return {'torso': torso.ravel(), 'fence': fence.ravel(), 'other': (~(torso | fence)).ravel()}


def read_references(run):
    refs = {}
    for arm in ('forward', 'reverse'):
        with np.load(Path(run)/arm/'reference_amf.npz', allow_pickle=False) as values:
            key = 'block_20_attn1_processor'
            flow, mask = values[key], values[key+'_mask']
            if flow.shape != (36, 1560, 2) or mask.shape != (36, 1560):
                raise ValueError('Unexpected saved reference shape')
            if not np.isfinite(flow).all() or mask.dtype != np.bool_:
                raise ValueError('Invalid reference flow or mask')
            pairs = np.arange(5)*7+1
            refs[arm] = (flow[pairs].copy(), mask[pairs].copy())
    return refs


def reference_statistics(refs, roi_masks):
    common = refs['forward'][1] & refs['reverse'][1]
    result = {}
    for support in ('own_valid', 'common_valid'):
        result[support] = {}
        for arm, (flow, valid) in refs.items():
            valid = valid if support == 'own_valid' else common
            result[support][arm] = {}
            for name, roi in roi_masks.items():
                rows = []
                for i in range(len(flow)):
                    mask = valid[i] & roi
                    x = flow[i, mask, 0]
                    rows.append(dict(pair=[i, i+1], roi_count=int(roi.sum()), valid_count=int(mask.sum()),
                        coverage=float(mask.sum()/roi.sum()),
                        original_valid_count=int((refs[arm][1][i] & roi).sum()),
                        zero_vector_count=int((roi & (flow[i] == 0).all(axis=-1)).sum()),
                        mean_xy=flow[i, mask].mean(axis=0).tolist() if mask.any() else None,
                        median_xy=np.median(flow[i, mask], axis=0).tolist() if mask.any() else None,
                        fraction_dx_positive=float((x > 0).mean()) if mask.any() else None,
                        fraction_dx_negative=float((x < 0).mean()) if mask.any() else None))
                result[support][arm][name] = rows
    return result


def inspect_saved_references(run, output):
    """CPU-only review that works on the small package BEFORE model loading."""
    run, output = Path(run), Path(output)
    refs = read_references(run)
    stats = reference_statistics(refs, regions())
    trace = {arm: json.loads((run/arm/'step_trace.json').read_text()) for arm in refs}
    result = dict(status='Completed saved-target inspection; this function executes zero transformer forwards',
        units='XY displacement in 16-pixel patches; positive dx is screen right, not camera-corrected travel',
        limitations='Fixed ROIs, not subject segmentation. Reverse pair i corresponds approximately to original pair 4-i; '
        'raw same-index comparisons measure the actual optimized objectives, not reversed physical-frame alignment. '
        'Review torso and fence separately. Coarse latent-frame mapping and camera motion prevent RGB-direction '
        'truth from being inferred from vector sign alone.',
        reference_statistics=stats,
        adjacent_mask_equals_nonzero_flow={a:bool(np.array_equal(m, np.any(f != 0, axis=-1))) for a,(f,m) in refs.items()},
        reference_rgb_region_statistics=reference_statistics(refs, regions(reference_rgb=True)),
        region_definitions=dict(objective_grid='Generated torso (350,140,590,260), fence (0,200,140,350); '
            'these are also the exact positions used for comparing target vectors in the loss.',
            reference_rgb='Reference torso (360,160,500,260), fence (20,120,180,190); '
            'separate conservative image-checked regions for interpreting reference video motion.'),
        recorded_update_magnitudes={a: {k:t[k] for k in ('update_fp32', 'update_after_input_cast',
            'recomputed_cfg_prediction', 'next_scheduler_state')} for a,t in trace.items()},
        source_sha256={str(p.relative_to(run)):digest(p) for a in refs for p in
                       (run/a/'reference_amf.npz', run/a/'step_trace.json')})
    output.mkdir(parents=True, exist_ok=True)
    (output/'saved_target_review.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def render_saved_reference_fields(run, output):
    """CPU-only overlays for reviewing actual targets; no model or synthetic motion."""
    import cv2
    from PIL import Image, ImageDraw
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    refs = read_references(run)
    common = refs['forward'][1] & refs['reverse'][1]
    panel = Image.new('RGB', (416*5, 280*2), 'white')
    for row, arm in enumerate(('forward', 'reverse')):
        cap = cv2.VideoCapture(str(run/arm/'original.mp4'))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if len(frames) != 21:
            raise ValueError('Expected 21 reference RGB frames')
        flow, valid = refs[arm]
        for pair in range(5):
            im = frames[pair*4].copy()
            roi = regions(reference_rgb=True)
            # Common=cyan, own-only=orange, invalid=gray. Draw sampled reference-ROI patches.
            for idx in np.flatnonzero(roi['torso'] | roi['fence']):
                y, x = divmod(int(idx), 52)
                if (x+y) % 2:
                    continue
                a = (x*16+8, y*16+8)
                color = (255, 255, 0) if common[pair, idx] else (0, 165, 255)
                if valid[pair, idx]:
                    delta = flow[pair, idx]*16
                    b = tuple(np.rint(np.asarray(a)+delta).astype(int))
                    cv2.arrowedLine(im, a, b, color, 1, tipLength=.25)
                else:
                    cv2.circle(im, a, 2, (128, 128, 128), -1)
            for box, color in (((360,160,500,260), (0,0,255)), ((20,120,180,190), (0,255,0))):
                cv2.rectangle(im, box[:2], box[2:], color, 2)
            rgb = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).resize((416, 240))
            panel.paste(rgb, (pair*416, row*280+40))
            ImageDraw.Draw(panel).text((pair*416+4, row*280+3),
                f'{arm} latent pair {pair}->{pair+1}\nRGB anchor {pair*4}; arrows 1x displacement', fill='black')
    panel.save(output/'reference_fields.jpg', quality=95)


def decompose(before, after, reference, valid, roi_masks):
    """Partition the actual global masked MSE; contributions sum to global loss."""
    before, after, reference = [np.asarray(v, dtype=np.float64) for v in (before, after, reference)]
    valid = np.asarray(valid, bool)
    if before.shape != after.shape or before.shape != reference.shape or before.shape[:-1] != valid.shape:
        raise ValueError('Flow/mask shapes differ')
    if before.shape[-1] != 2 or not valid.any():
        raise ValueError('Expected XY flow and nonempty validity mask')
    if not all(np.isfinite(v).all() for v in (before, after, reference)):
        raise ValueError('Nonfinite flow')
    partition = np.stack([np.broadcast_to(m, valid.shape) for m in roi_masks.values()])
    if not np.all(partition.sum(axis=0) == 1):
        raise ValueError('Regions must be a disjoint exhaustive partition')
    e0, e1 = ((before-reference)**2).mean(axis=-1), ((after-reference)**2).mean(axis=-1)
    denominator = int(valid.sum())
    result = {'loss_before': float(e0[valid].mean()), 'loss_after': float(e1[valid].mean()), 'regions': {}}
    for name, roi in roi_masks.items():
        mask = valid & roi
        n = int(mask.sum())
        result['regions'][name] = dict(valid_count=n, valid_fraction=n/denominator,
            mean_mse_before=float(e0[mask].mean()) if n else None,
            mean_mse_after=float(e1[mask].mean()) if n else None,
            contribution_before=float(e0[mask].sum()/denominator),
            contribution_after=float(e1[mask].sum()/denominator),
            reduction_contribution=float((e0[mask]-e1[mask]).sum()/denominator),
            pairs=[dict(valid_count=int(m.sum()),
                        before_xy=before[i][m].mean(axis=0).tolist() if m.any() else None,
                        after_xy=after[i][m].mean(axis=0).tolist() if m.any() else None,
                        reference_xy=reference[i][m].mean(axis=0).tolist() if m.any() else None)
                   for i, m in enumerate(mask)])
    return result


def load_inputs(run):
    run = Path(run)
    manifest = json.loads((run/'pilot_manifest.json').read_text())
    exp = manifest['experiment']
    if (exp['guidance_blocks'] != [20] or exp['token_grid'] != [6, 30, 52]
            or exp['guidance_index'] != 39 or exp['target_readout'] != 'centered logits, multiplier 8'
            or exp['injection']):
        raise ValueError('This diagnostic is specific to the reviewed centered camel pilot')
    files = [run/'pilot_manifest.json', run/'trace_gate.json']
    latents, refs, traces = {}, read_references(run), {}
    for arm in ('forward', 'reverse'):
        trace_path = run/arm/'step_trace.json'
        latent_path = run/arm/'step_trace.npz'
        ref_path = run/arm/'reference_amf.npz'
        files.extend((trace_path, latent_path, ref_path))
        traces[arm] = json.loads(trace_path.read_text())
        if not traces[arm]['passed']:
            raise ValueError('Requires completed real-step propagation evidence')
        with np.load(latent_path, allow_pickle=False) as values:
            before, after = values['before'].copy(), values['after'].copy()
        if before.shape != (1, 16, 6, 60, 104) or after.shape != before.shape:
            raise ValueError('Unexpected saved latent dimensions')
        if before.dtype != np.float32 or after.dtype != np.float32:
            raise ValueError('Expected preserved FP32 latents')
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError('Nonfinite saved latent')
        if 'before' in latents and not np.array_equal(latents['before'], before):
            raise ValueError('Arms did not start from identical latents')
        latents['before'], latents[arm] = before, after
    return manifest, latents, refs, traces, {str(p.relative_to(run)): digest(p) for p in files}


def capture(g, latent, index=39):
    """Exactly the pilot's target readout, without optimizer, CFG or scheduler calls."""
    import torch
    from benchmark.wan_centered_pilot import clear
    from guidance_utils.wan_centered_amf import centered_pair_flow
    clear(g)
    try:
        with torch.no_grad():
            g._set_kv_mode([20], inject=False, copy=True)
            x = torch.from_numpy(latent).to(g.device)
            g._forward_transformer(x, g.guidance_embeds[1:2], g.timesteps[index].expand(x.shape[0]))
            processor = g.transformer.blocks[20].attn1.processor
            if processor.query.shape != (1, 9360, 40, 128) or processor.key.shape != processor.query.shape:
                raise ValueError('Expected native B,S,H,D normalized post-RoPE Q/K')
            q, k = [v[0].reshape(6, 1560, 40, 128) for v in (processor.query, processor.key)]
            flow = torch.stack([centered_pair_flow(q[i], k[i+1], 30, 52)
                                for i in range(5)]).float().cpu().numpy()
            queries = mass_queries()
            mass = native_frame_mass(processor.query[0], processor.key[0],
                                     [q['token_index'] for q in queries], 6, 1560)
            return flow, mass
    finally:
        clear(g)


def inspect_saved_states(g, run, output):
    """Use an already loaded frozen pilot model; never rerun trace_pilot/decode_pilot."""
    from benchmark.wan_runtime_preflight import verify_runtime
    from benchmark.wan_source_fingerprints import verify_parity_sources
    from probe_wan_response import tensor_hash
    run, output = Path(run), Path(output)
    saved_review = inspect_saved_references(run, output)
    manifest, latents, refs, traces, inputs = load_inputs(run)
    baseline = manifest['baseline']
    root = Path(__file__).resolve().parents[1]
    runtime = verify_runtime({**baseline['packages'], 'tokenizers': '0.22.2'})
    sources = verify_parity_sources(root, baseline['source_sha256'])
    if digest(root/'guidance_utils/wan_centered_amf.py') != manifest['kernel_sha256']:
        raise ValueError('Centered kernel changed')
    if baseline['checkpoint_revision'] not in str(g.config.model_key):
        raise ValueError('Expected the pinned snapshot path')
    actual = dict(initial_latent_sha256=tensor_hash(g.init_latents),
                  conditioning_sha256=tensor_hash(g.guidance_embeds),
                  source_conditioning_sha256=tensor_hash(g.source_embeds),
                  rope_sha256=tensor_hash(g.transformer.init_rope))
    if actual != manifest['checks'] or list(g.config.guidance_blocks) != [20]:
        raise ValueError('Live model conditioning/noise/RoPE/blocks differ')
    if g.timesteps.cpu().tolist() != baseline['timesteps'] or g.scheduler.sigmas.tolist() != baseline['sigmas']:
        raise ValueError('Live schedule differs')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='The aggregate loss reduction may hide unhelpful subject correspondences or background improvement.',
        expected='Regional loss accounting, common-mask target responses, and native destination-frame mass from the SAME Q/K captures.',
        max_positive_forwards=3, max_optimization_updates=0, max_sampling_steps=0, max_decodes=0,
        inputs_sha256=inputs, runtime=runtime, core_sources=sources,
        diagnostic_sha256=digest(__file__), masks='fixed conservative torso/fence ROIs; other is NOT all background',
        native_attention_queries=mass_queries(),
        native_attention_method='Per-head FP32 QK/sqrt(head_dim), all 9360 valid keys, no centering/sharpening/head averaging. '
        'Formula reconstruction, not bitwise fused-SDPA weights. Low mass is a clue, not causal proof.')
    marker = output/'started.json'
    with marker.open('x') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: 3 positive forwards; zero optimizer, scheduler or VAE calls.', flush=True)
    fields, masses = {}, {}
    for name in ('before', 'forward', 'reverse'):
        print(f'Reading saved {name} latent', flush=True)
        fields[name], masses[name] = capture(g, latents[name])
        np.savez_compressed(output/f'{name}_flow.npz', flow=fields[name])
        np.savez_compressed(output/f'{name}_native_mass.npz', mass=masses[name])
    common = refs['forward'][1] & refs['reverse'][1]
    report = dict(port_success=False, analysis='Regional loss accounting, not spatial attribution of the latent update',
        roi_limitations='Torso and fence are conservative samples, not full segmentation; '
        'other contains limbs and background. Centering couples all source queries; lower torso-indexed '
        'loss does not establish localized torso intervention. Inspect fields, not just means.',
        saved_target_review=saved_review, arms={},
        native_attention=dict(queries=mass_queries(), states={name:summarize_mass(mass, mass_queries()) for name,mass in masses.items()},
            changes={name:summarize_mass(masses[name]-masses['before'], mass_queries()) for name in ('forward', 'reverse')},
            interpretation='Mass is diagnostic only. No universal low-mass cutoff; values, output projection and later layers also matter.'))
    for arm in ('forward', 'reverse'):
        ref, valid = refs[arm]
        result = decompose(fields['before'], fields[arm], ref, valid, regions())
        expected = [traces[arm]['losses_before_updates'][0], traces[arm]['loss_after_updates']]
        observed = [result['loss_before'], result['loss_after']]
        result['loss_replay'] = dict(recorded=expected, recomputed=observed, rtol=.005, atol=.005,
                                     passed=bool(np.allclose(observed, expected, rtol=.005, atol=.005)))
        result['common_valid_loss_accounting'] = (decompose(fields['before'], fields[arm], ref, common, regions())
                                                  if common.any() else None)
        result['common_valid_response'] = response_to_target(fields['before'], fields[arm], ref, common, regions())
        report['arms'][arm] = result
    report['loss_replay_passed'] = all(a['loss_replay']['passed'] for a in report['arms'].values())
    (output/'attribution.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['loss_replay_passed']:
        raise RuntimeError('Saved AMF losses did not replay within 0.5% + 0.005. Inspect reports; do not repeat capture.')
    print('Saved attribution. This is objective diagnosis, not motion-transfer proof.', flush=True)
    return report


def response_to_target(before, after, target, common, roi_masks):
    """Per-pair response relative to the before state on identical positions in both arms."""
    delta = after.astype(np.float64)-before
    desired = target.astype(np.float64)-before
    result = {}
    for name, roi in roi_masks.items():
        result[name] = []
        for i in range(len(before)):
            mask = common[i] & roi
            change, need = delta[i, mask], desired[i, mask]
            norm = float(np.linalg.norm(change)*np.linalg.norm(need))
            result[name].append(dict(pair=[i, i+1], valid_count=int(mask.sum()),
                delta_xy=change.mean(axis=0).tolist() if mask.any() else None,
                desired_delta_xy=need.mean(axis=0).tolist() if mask.any() else None,
                delta_rms=float(np.sqrt((change**2).mean())) if mask.any() else None,
                target_error_rms=float(np.sqrt((need**2).mean())) if mask.any() else None,
                patchwise_alignment_cosine=float((change*need).sum()/norm) if norm else None))
    return result
