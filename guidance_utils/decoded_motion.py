"""Motion measured in decoded video: the acceptance metric for the decisive Wan test.

Every earlier Wan diagnostic scored AMF inside the model. Those scores improved while the
decoded videos stayed the same. This module measures motion in the pixels instead:
Farneback optical flow between frames four apart (one latent frame), pooled to the 30x52
patch grid so it is expressed in the same patch units as the AMF loss.

The decisive quantity is direction selectivity. Guidance toward a reference and guidance
toward the time-reversed reference must push the decoded motion apart, along the
difference between the two references' own decoded motion. A generic perturbation of the
same size moves the videos too, but not along that direction.
"""
import math

import numpy as np

GRID = (30, 52)          # Wan 480x832 latent patch grid, (rows, columns)
STRIDE = 4               # decoded frames per latent frame
FRAMES = 21
SIZE = (832, 480)        # (width, height) of every decoded video
WORK = (208, 120)        # flow working resolution: exactly 4 px per patch
FARNEBACK = (.5, 4, 21, 5, 7, 1.5, 0)

# Fixed before any decisive run. Changing them requires a fresh plan.
THRESHOLDS = dict(
    positive_control_min=0.5,   # SDEdit forward/reverse must separate at least this well
    amf_min=0.3,                # AMF forward/reverse selectivity needed to pass
    margin_over_null=0.2,       # ...and by this much more than two equal-dose random arms
    min_reference_motion=0.1,   # patches per latent frame; a static reference cannot test direction
)


def read_frames(path, count=FRAMES, size=SIZE):
    import cv2
    import imageio.v3 as iio
    frames = []
    for frame in iio.imiter(path, plugin='FFMPEG'):
        frames.append(cv2.resize(np.asarray(frame)[..., :3], size, interpolation=cv2.INTER_AREA))
        if len(frames) == count:
            break
    if len(frames) != count:
        raise ValueError(f'{path}: expected {count} frames, found {len(frames)}')
    return np.stack(frames)


def flow_field(frames, stride=STRIDE, grid=GRID):
    """(pairs, rows, columns, 2) displacement in patch units, x right and y down."""
    import cv2
    rows, columns = grid
    scale = WORK[0] / columns
    if WORK[1] / rows != scale:
        raise ValueError('Working resolution must have square patches')
    gray = [cv2.cvtColor(cv2.resize(np.asarray(f, dtype=np.uint8), WORK, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY) for f in frames]
    fields = [cv2.resize(cv2.calcOpticalFlowFarneback(gray[i], gray[i + stride], None, *FARNEBACK),
                         (columns, rows), interpolation=cv2.INTER_AREA) / scale
              for i in range(0, len(gray) - stride, stride)]
    if not fields:
        raise ValueError('Too few frames for one flow pair')
    return np.stack(fields).astype(np.float64)


def _cosine(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float('nan')


def selectivity(guided_forward, guided_reverse, reference_forward, reference_reverse):
    """How far two arms separate along the direction their two references separate.

    score: cosine between (forward arm - reverse arm) and (forward reference - reverse reference);
           +1 steered exactly toward the references, 0 unrelated, nan when the arms are identical.
    gain:  projection of that separation onto the reference separation; 1 means full transfer.
    """
    delta = np.asarray(guided_forward, float) - guided_reverse
    target = np.asarray(reference_forward, float) - reference_reverse
    energy = float((target * target).sum())
    return dict(score=_cosine(delta, target),
                gain=float((delta * target).sum() / energy) if energy > 1e-12 else float('nan'),
                separation=float(np.linalg.norm(delta, axis=-1).mean()),
                target_magnitude=float(np.linalg.norm(target, axis=-1).mean()))


def change_from(arm, base, reference):
    """Motion one arm added relative to another, and whether it points along a reference."""
    delta = np.asarray(arm, float) - base
    return dict(alignment=_cosine(delta, reference), magnitude=float(np.linalg.norm(delta, axis=-1).mean()))


def verdict(amf, null, positive, thresholds=THRESHOLDS):
    """GO / STOP / INVALID from the three selectivity results. Pure; no model or file access."""
    t = thresholds
    s, n, p = amf['score'], null['score'], positive['score']
    null_size = abs(n) if math.isfinite(n) else 0.0
    base = dict(amf_score=s, null_score=n, positive_score=p, null_size=null_size, thresholds=dict(t))
    if positive['target_magnitude'] < t['min_reference_motion']:
        return dict(base, decision='INVALID', reasons=['The two references barely move, so direction cannot be tested.'])
    if not math.isfinite(p) or p < t['positive_control_min']:
        return dict(base, decision='INVALID', reasons=[
            f"Positive control separated the references at {p:.3f}, below {t['positive_control_min']}: "
            'the measurement cannot see steering in this pipeline.'])
    reasons = []
    if not math.isfinite(s):
        reasons.append('The forward and reverse AMF videos have identical decoded motion.')
    else:
        if s < t['amf_min']:
            reasons.append(f"AMF selectivity {s:.3f} is below {t['amf_min']}.")
        if s - null_size < t['margin_over_null']:
            reasons.append(f"AMF selectivity {s:.3f} does not beat the equal-dose random pair ({null_size:.3f}) "
                           f"by {t['margin_over_null']}.")
    return dict(base, decision='STOP' if reasons else 'GO', reasons=reasons or [
        f"AMF selectivity {s:.3f} passes both the absolute threshold and the random-pair margin."])
