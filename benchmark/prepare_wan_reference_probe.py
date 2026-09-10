"""Package exact pilot frames plus DAVIS masks for an offline reference audit."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

CASES = ('car-turn', 'camel')


def prepare(pilot_inputs, davis, output):
    pilot_inputs, davis, output = map(Path, (pilot_inputs, davis, output))
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    with (pilot_inputs / 'manifest.csv').open(newline='', encoding='utf-8') as handle:
        manifest = list(csv.DictReader(handle))
    selections = []
    for clip in CASES:
        rows = [r for r in manifest if r['clip_id'] == clip and r['prompt_id'] == 'subject']
        if len(rows) != 1:
            raise ValueError(f'Expected one subject row: {clip}')
        row = rows[0]
        source = pilot_inputs / row['video_path']
        frames = sorted(source.glob('*.jpg'))
        digest = hashlib.sha256()
        for frame in frames:
            digest.update(frame.read_bytes())
        if len(frames) != 24 or digest.hexdigest() != row['sha256']:
            raise ValueError(f'Changed pilot input: {clip}')
        packing = json.loads((source / 'meta.json').read_text(encoding='utf-8'))
        masks = [davis / 'Annotations/480p' / clip / f'{i:05d}.png' for i in packing['indices'][:21]]
        if len(masks) != 21 or not all(p.is_file() for p in masks):
            raise FileNotFoundError(f'Missing DAVIS masks: {clip}')
        selections.append((row, source, frames, masks, packing))
    output.mkdir(parents=True)
    files = {}
    rows = []
    for row, source, frames, masks, packing in selections:
        clip = row['clip_id']
        image_dir, mask_dir = output / 'clips' / clip, output / 'masks' / clip
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for frame in frames:
            shutil.copy2(frame, image_dir / frame.name)
        shutil.copy2(source / 'meta.json', image_dir / 'meta.json')
        for i, mask in enumerate(masks):
            shutil.copy2(mask, mask_dir / f'{i:05d}.png')
        rows.append({**row, 'video_path': f'clips/{clip}', 'mask_path': f'masks/{clip}'})
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'checksums.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    (output / 'README.txt').write_text(
        'Exact existing pilot JPEGs; first 21 consumed. Masks are copied from the corresponding\n'
        'DAVIS source indices in each clip meta.json and renamed to packed frame indices.\n'
        'Masks are for OFFLINE DIAGNOSTICS ONLY; they are not input to Wan or its guidance.\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent).as_posix())
    print(f'Prepared {len(rows)} references with aligned masks: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-inputs', type=Path, default=Path('probe_runs/wan_davis_pilot_inputs'))
    parser.add_argument('--davis', type=Path, default=Path('DAVIS'))
    parser.add_argument('--output', type=Path, default=Path('probe_runs/wan_reference_inputs'))
    args = parser.parse_args()
    prepare(args.pilot_inputs, args.davis, args.output)
