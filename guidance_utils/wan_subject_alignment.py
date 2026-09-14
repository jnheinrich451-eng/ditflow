"""Experimental coarse subject remapping, isolated from production Wan defaults."""
import numpy as np
import torch


def aligned_targets(reference, valid, regions, reference_boxes, generated_boxes, image_size):
    """Remap subject samples with the SAME source-frame transform at both endpoints.

    Boxes use pixel-edge xyxy coordinates; fields use patch displacement units.
    For query y and source-frame map T_i(p)=S_i*p+b_i, use S_i*f(T_i^-1(y)).
    Using T_j for the endpoint would insert the baseline box trajectory and could
    cancel the reference motion. Nearest patch sampling preserves hard targets.
    Background keeps screen coordinates, excluding both subject locations.
    """
    reference = np.asarray(reference, dtype=np.float32)
    valid, regions = np.asarray(valid, dtype=bool), np.asarray(regions, dtype=bool)
    frames, height, width = regions.shape
    shape = (frames*frames, height*width, 2)
    if reference.shape != shape or valid.shape != shape[:-1] or not np.isfinite(reference).all():
        raise ValueError('Invalid AMF field/mask geometry')
    boxes = [np.asarray(b, dtype=float) for b in (reference_boxes, generated_boxes)]
    pixel_width, pixel_height = image_size
    for box in boxes:
        if (box.shape != (frames, 4) or not np.isfinite(box).all() or np.any(box[:, 2:] <= box[:, :2])
                or np.any(box < 0) or np.any(box[:, [0, 2]] > pixel_width)
                or np.any(box[:, [1, 3]] > pixel_height)):
            raise ValueError('Invalid subject boxes')
    reference_boxes, generated_boxes = boxes
    yy, xx = np.indices((height, width))
    points = np.stack(((xx+.5)*pixel_width/width, (yy+.5)*pixel_height/height), axis=-1)
    flow = reference.copy()
    subject = np.zeros(shape[:-1], dtype=bool)
    background = np.zeros_like(subject)
    source_index = np.full(shape[:-1], -1, dtype=np.int32)
    scales = (generated_boxes[:, 2:]-generated_boxes[:, :2])/(reference_boxes[:, 2:]-reference_boxes[:, :2])
    for i in range(frames-1):
        pair = i*frames+i+1
        rb, gb, scale = reference_boxes[i], generated_boxes[i], scales[i]
        source = (points-gb[:2])/scale+rb[:2]
        sx = np.floor(source[..., 0]*width/pixel_width).astype(int)
        sy = np.floor(source[..., 1]*height/pixel_height).astype(int)
        inside = ((sx >= 0) & (sx < width) & (sy >= 0) & (sy < height))
        generated_region = ((points[..., 0] >= gb[0]) & (points[..., 0] < gb[2]) &
                            (points[..., 1] >= gb[1]) & (points[..., 1] < gb[3]))
        index = np.clip(sy, 0, height-1)*width+np.clip(sx, 0, width-1)
        fg = inside & generated_region & regions[i].reshape(-1)[index] & valid[pair][index]
        bg = valid[pair].reshape(height, width) & ~regions[i] & ~generated_region
        subject[pair] = fg.ravel(); background[pair] = bg.ravel()
        source_index[pair, fg.ravel()] = index[fg]
        flow[pair, fg.ravel()] = reference[pair, index[fg]]*scale
    mask = subject | background
    flow[~mask] = 0
    if not subject.any() or not background.any():
        raise ValueError('Alignment requires retained subject and background entries')
    return dict(flow=flow, mask=mask, subject=subject, background=background,
                source_index=source_index, scales=scales.astype(np.float32),
                reference_boxes=reference_boxes.astype(np.float32), generated_boxes=generated_boxes.astype(np.float32))


def alignment_loss(prediction, reference, subject, background, mode):
    """Uniform retained-entry MSE or equal means of the two disjoint regions."""
    if (prediction.shape != reference.shape or prediction.shape[:-1] != subject.shape
            or subject.shape != background.shape or subject.dtype != torch.bool or background.dtype != torch.bool
            or torch.any(subject & background) or not subject.any() or not background.any()):
        raise ValueError('Invalid or empty aligned regions')
    if mode not in ('uniform', 'balanced'):
        raise ValueError('Unknown alignment loss mode')
    error = (prediction-reference).square().mean(-1)
    subject_loss, background_loss = error[subject].mean(), error[background].mean()
    weight = .5 if mode == 'balanced' else float(subject.sum())/float((subject | background).sum())
    total = (torch.nn.functional.mse_loss(prediction[subject | background], reference[subject | background])
             if mode == 'uniform' else .5*subject_loss+.5*background_loss)
    return total, subject_loss, background_loss, weight
