"""Package six DAVIS subject-prompt cases without changing benchmark frames.

python benchmark/prepare_wan_pilot.py --packed-root E:/bench/packed/davis \
    --out probe_runs/wan_davis_pilot_inputs

The ZIP includes its root folder, manifest, original 24 packed JPEGs per clip,
and source metadata. Wan will consume the first 21 packed frames. No recutting
or image re-encoding is performed; every clip must match davis50.csv's hash.
"""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = {
    'blackswan': 'Smooth translation on water; object direction and background stability.',
    'camel': 'Slow quadruped walking; sustained gait and camera tracking.',
    'car-turn': 'Turning vehicle; changing heading and apparent size.',
    'dance-twirl': 'Body rotation; pose sequence and sustained articulation.',
    'horsejump-low': 'Jumping; takeoff, clearance, landing timing and tracking.',
    'dogs-jump': 'Multiple moving subjects; separate trajectories and timing.',
}


def prepare(manifest, packed_root, out):
    with Path(manifest).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for clip_id in CASES:
        matches = [r for r in rows if r['clip_id'] == clip_id and r['prompt_id'] == 'subject']
        if len(matches) != 1:
            raise ValueError(f'Expected one subject prompt for {clip_id}; found {len(matches)}')
        row = matches[0]
        src = Path(packed_root) / clip_id
        files = sorted(src.glob('*.jpg'))
        if len(files) != 24:
            raise ValueError(f'{src}: expected 24 packed JPEG frames, found {len(files)}')
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.read_bytes())
        if digest.hexdigest() != row['sha256']:
            raise ValueError(f'{clip_id}: packed frame bytes differ from the benchmark manifest')
        selected.append((row, src, files))
    out = Path(out).resolve()
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise FileExistsError(f'Use a fresh output name: {out} or {archive} already exists')
    out.mkdir(parents=True)
    portable = []
    for row, src, files in selected:
        dest = out / 'clips' / row['clip_id']
        dest.mkdir(parents=True)
        for file in files:
            shutil.copy2(file, dest / file.name)
        if (src / 'meta.json').exists():
            shutil.copy2(src / 'meta.json', dest / 'meta.json')
        portable.append({**row, 'video_path': f'clips/{row["clip_id"]}',
                         'pilot_focus': CASES[row['clip_id']]})
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(portable[0]))
        writer.writeheader()
        writer.writerows(portable)
    (out / 'selection.json').write_text(json.dumps({
        'purpose': 'Wan port development pilot, not a held-out performance estimate',
        'source_manifest': str(Path(manifest)), 'cases': CASES,
        'packed_frames': 24, 'wan_consumed_frames': 21,
        'prompt_selection': 'prompt_id=subject; original prompt text unchanged',
    }, indent=2), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out.parent).as_posix())
    print(f'{len(portable)} subject prompts; all six clip hashes verified')
    print(f'Input ZIP: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)')
    return out, archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO / 'benchmark/davis50.csv')
    parser.add_argument('--packed-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=REPO / 'probe_runs/wan_davis_pilot_inputs')
    args = parser.parse_args()
    prepare(args.manifest, args.packed_root, args.out)
