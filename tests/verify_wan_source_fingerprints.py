"""Weight-free regression: legacy mixed EOL -> Git LF; reject real edits."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.wan_source_fingerprints import AUDITED_MIXED_EOL, CORE_SOURCES, digest, verify_source, verify_parity_sources


def reject(name, raw, expected):
    try:
        verify_source(name, raw, expected)
    except ValueError:
        return
    raise AssertionError(f'Unexpected acceptance: {name}')


def main():
    for (name, expected), canonical in AUDITED_MIXED_EOL.items():
        # Git returns LF bytes on both Windows and Linux: reproduce the Colab checkout.
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + name])
        assert digest(blob) == canonical, name
        for data in (blob, blob.replace(b'\n', b'\r\n')):
            result = verify_source(name, data, expected)
            assert result['method'] == 'audited_mixed_line_endings'
        assert verify_source(name, (ROOT/name).read_bytes(), expected)
        reject(name, blob + b'\nactual_change = True\n', expected)
        reject(name, blob + b'# comment-only edits are not line endings\n', expected)
        reject(name, blob.replace(b' ', b'\t', 1), expected)
        reject(name, blob, '0' * 64)
        reject('wrong/path.py', blob, expected)
    for raw in (b'a\nb\n', b'a\r\nb\r\n', b'a\r\nb\n'):
        assert verify_source('fixture.py', raw, digest(raw))['method'] == 'exact'
    assert verify_source('fixture.py', b'a\nb\n', digest(b'a\r\nb\r\n'))['method'] == 'uniform_line_endings'
    reject('fixture.py', b'a\nb\n', digest(b'a\r\nb\n'))  # unaudited mixed hash must fail
    # Optional retained artifact integration; the pure Git fixtures above always run.
    manifest = ROOT/'probe_runs/wan_port_acceptance_camel_s1_v2_retry/manifest.json'
    if manifest.exists():
        recorded = json.loads(manifest.read_text())['source_sha256']
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory)
            for name in CORE_SOURCES:
                path = clone / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + name]))
            assert len(verify_parity_sources(clone, recorded)) == 7
            path = clone / 'motion_guidance_wan.py'
            path.write_bytes(path.read_bytes().replace(b'class WanGuidance(', b'class AlteredWanGuidance(', 1))
            try:
                verify_parity_sources(clone, recorded)
            except ValueError:
                pass
            else:
                raise AssertionError('Altered cloned implementation was accepted')
    print('PASS: audited mixed endings and all seven Linux-clone fingerprints; code/comment/whitespace edits and unknown hashes rejected. No GPU or model calls.')


if __name__ == '__main__':
    main()
