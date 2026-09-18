"""Offline decoded-pixel comparison; never imports the model or runs sampling."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f'No frames: {path}')
    return np.stack(frames)


def track_points(frames, initial):
    """Track explicitly chosen RGB landmarks, with consecutive forward/backward checks."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    points = np.asarray(initial, np.float32).reshape(-1, 1, 2)
    tracks = [points[:, 0].copy()]
    errors, statuses = [], []
    params = dict(winSize=(25, 25), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001))
    for a, b in zip(gray[:-1], gray[1:]):
        nxt, valid, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **params)
        back, valid_back, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **params)
        errors.append(np.linalg.norm(back[:, 0] - points[:, 0], axis=1))
        statuses.append((valid[:, 0] & valid_back[:, 0]).astype(bool))
        tracks.append(nxt[:, 0].copy())
        points = nxt
    return np.stack(tracks), np.stack(errors), np.stack(statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run
    out = root / 'review'
    out.mkdir(exist_ok=True)
    paths = {a: root / a / 'final.mp4' for a in ('off', 'forward', 'reverse')}
    paths.update({a + '_reference': root / a / 'original.mp4' for a in ('forward', 'reverse')})
    videos = {a: read_video(p) for a, p in paths.items()}
    report = dict(method='OpenCV decoded RGB uint8; errors in 0..255 channel levels; lossy MP4 evidence',
                  files={}, comparisons={}, port_success=False)
    for name, frames in videos.items():
        report['files'][name] = dict(shape=list(frames.shape),
            file_sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            decoded_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
    for a, b in itertools.combinations(('off', 'forward', 'reverse'), 2):
        x, y = videos[a], videos[b]
        if x.shape != y.shape:
            raise ValueError('Generated frame shapes differ')
        delta = np.abs(x.astype(np.float32) - y.astype(np.float32))
        report['comparisons'][a + '_vs_' + b] = dict(
            identical=bool(np.array_equal(x, y)),
            identical_frames=[i for i in range(len(x)) if np.array_equal(x[i], y[i])],
            mean_absolute_error=float(delta.mean()), rms=float(np.sqrt(np.mean(delta ** 2))),
            excluding_frame0_mae=float(delta[1:].mean()),
            changed_pixel_fraction=float(np.any(delta != 0, axis=-1).mean()),
            per_frame_mae=delta.mean(axis=(1, 2, 3)).tolist())
    indices = (1, 4, 8, 12, 16, 20)
    sheet = Image.new('RGB', (416 * len(indices), 268 * len(videos)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (name, frames) in enumerate(videos.items()):
        for col, i in enumerate(indices):
            sheet.paste(Image.fromarray(frames[i]).resize((416, 240)), (col*416, row*268+28))
            draw.text((col*416+5, row*268+5), f'{name} | frame {i}', fill='black')
    sheet.save(out / 'contact_sheet.jpg', quality=95)
    for name, frames in videos.items():
        Image.fromarray(frames[1]).save(out / f'{name}_frame01.png')
        Image.fromarray(frames[20]).save(out / f'{name}_frame20.png')
    # These points are selected visually in this camel case; this is not a generic detector.
    # Point 0: rear hump apex. Points 1..3: visible stationary fence features.
    # Reference runs backwards for the reversed clip, so it starts in matching geometry.
    generated_points = [(506, 58), (88, 356), (96, 211), (212, 235)]
    reference_points = [(356, 92), (420, 75), (508, 68), (253, 182)]
    tracking = dict(method='Pyramidal Lucas-Kanade on manually selected decoded RGB landmarks; '
        'frame 0 excluded; subject hump displacement minus median of three fence displacements. '
        'Tracking confidence is a diagnostic, not semantic validation; inspect the overlays. '
        'Reject fence tracks with any forward/backward error >1 pixel. Reference hump LK '
        'tracks were rejected on visual review: they drift onto background despite sometimes low error.',
        anchors_generated=generated_points, anchors_reference=reference_points, videos={})
    track_sheet = Image.new('RGB', sheet.size, 'white')
    track_draw = ImageDraw.Draw(track_sheet)
    for row, (name, frames) in enumerate(videos.items()):
        reverse = name == 'reverse_reference'
        selected = frames[1:][::-1] if reverse else frames[1:]
        initial = reference_points if 'reference' in name else generated_points
        tracks, errors, statuses = track_points(selected, initial)
        if reverse:
            tracks = tracks[::-1]
        displacement = tracks[-1] - tracks[0]
        good_fences = (errors.max(axis=0)[1:] < 1) & statuses.all(axis=0)[1:]
        if good_fences.sum() < 2:
            raise ValueError(f'{name}: fewer than two reliable fence points')
        fence_dx = float(np.median(displacement[1:, 0][good_fences]))
        is_reference = 'reference' in name
        tracking['videos'][name] = dict(
            subject_dx_px=float(displacement[0, 0]), fence_dx_px=displacement[1:, 0].tolist(),
            accepted_fence_ids=(np.flatnonzero(good_fences)+1).tolist(), median_accepted_fence_dx_px=fence_dx,
            subject_lk_accepted=not is_reference,
            subject_minus_median_fence_dx_px=None if is_reference else float(displacement[0, 0]-fence_dx),
            max_forward_backward_error_px=errors.max(axis=0).tolist(),
            all_tracking_status_ok=bool(statuses.all()),
            frame_indices=list(range(1, len(frames))), tracks_xy=tracks.tolist())
        if is_reference:
            endpoints = ([(334, 102), (355, 87)] if reverse else [(354, 88), (341, 108)])
            tracking['videos'][name]['manual_subject_endpoints_frame1_frame20_xy'] = endpoints
            tracking['videos'][name]['manual_subject_minus_fence_dx_px'] = endpoints[1][0]-endpoints[0][0]-fence_dx
            tracking['videos'][name]['manual_review_allowance_px'] = 32
            endpoint_sheet = Image.new('RGB', (1664, 508), 'white')
            for column, (fi, xy) in enumerate(zip((1, 20), endpoints)):
                im = Image.fromarray(frames[fi])
                d = ImageDraw.Draw(im)
                x, y = xy
                d.ellipse((x-8, y-8, x+8, y+8), outline='red', width=3)
                for x, y in tracks[fi-1, 1:]:
                    d.ellipse((x-6, y-6, x+6, y+6), outline='cyan', width=3)
                endpoint_sheet.paste(im, (column*832, 28))
                ImageDraw.Draw(endpoint_sheet).text((column*832+5, 5),
                    f'{name} frame {fi}: manual hump (red), tracked fence (cyan)', fill='black')
            endpoint_sheet.save(out / f'{name}_manual_endpoints.jpg', quality=95)
        for col, i in enumerate(indices):
            im = Image.fromarray(frames[i])
            d = ImageDraw.Draw(im)
            for j, (x, y) in enumerate(tracks[i-1]):
                accepted = (not is_reference) if j == 0 else good_fences[j-1]
                color = ('red' if j == 0 else 'cyan') if accepted else 'gray'
                d.ellipse((x-6, y-6, x+6, y+6), outline=color, width=3)
                d.text((x+8, y), str(j) + ('' if accepted else ' rejected'), fill=color)
            track_sheet.paste(im.resize((416, 240)), (col*416, row*268+28))
            track_draw.text((col*416+5, row*268+5), f'{name} | frame {i}', fill='black')
    track_sheet.save(out / 'landmark_tracks.jpg', quality=95)
    (out / 'landmark_tracking.json').write_text(json.dumps(tracking, indent=2) + '\n')
    (out / 'pixel_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['comparisons'], indent=2))
    print(json.dumps({k: {m: v for m, v in d.items() if m not in ('tracks_xy', 'frame_indices')}
                      for k, d in tracking['videos'].items()}, indent=2))


if __name__ == '__main__':
    main()
