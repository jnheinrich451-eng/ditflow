"""Local restoration preserves results and rejects unsafe/incompatible ZIPs."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.wan_archive_inputs import restore_previous_archive


def make_archive(path, prefix='run/', extra=None):
    files = {'plan.json': json.dumps({'stage': 'noised_reference_step9_visual'}),
             'plan.sha256': 'saved-plan-digest', 'environment.json': '{}',
             'confirmation_done.json': '{}', 'off_done.json': '{}', 'candidate_done.json': '{}',
             'confirmation/static/clean/block_30_attn1_processor/mean_logits.npz': 'saved-capture'}
    with zipfile.ZipFile(path, 'w') as bundle:
        for name, value in files.items(): bundle.writestr(prefix+name, value)
        if extra: bundle.writestr(*extra)


class ArchiveTests(unittest.TestCase):
    def test_nested_and_flat_archives_restore_capture_and_are_idempotent(self):
        for prefix in ('run/', ''):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); archive = root/'results.zip'; destination = root/'local/run'
                make_archive(archive, prefix)
                restored = restore_previous_archive(archive, destination)
                self.assertEqual(restored, destination)
                capture = restored/'confirmation/static/clean/block_30_attn1_processor/mean_logits.npz'
                self.assertEqual(capture.read_text(), 'saved-capture')
                self.assertEqual(restore_previous_archive(archive, destination), restored)

    def test_incomplete_existing_folder_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = root/'results.zip'; destination = root/'run'
            make_archive(archive); destination.mkdir(); (destination/'user-note').write_text('keep')
            restored = restore_previous_archive(archive, destination)
            self.assertNotEqual(restored, destination)
            self.assertEqual((destination/'user-note').read_text(), 'keep')
            self.assertEqual(restore_previous_archive(archive, destination), restored)

    def test_input_bundle_is_not_accepted_as_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = root/'inputs.zip'
            with zipfile.ZipFile(archive, 'w') as bundle: bundle.writestr('manifest.csv', 'clip_id')
            with self.assertRaisesRegex(ValueError, 'CONTROL_INPUT_ZIP'):
                restore_previous_archive(archive, root/'local')
            self.assertFalse((root/'local').exists())

    def test_unsafe_archive_paths_and_symlinks_are_rejected(self):
        names = ['../outside', '/absolute', 'C:/outside', 'run/../../outside']
        link = zipfile.ZipInfo('run/link'); link.create_system = 3; link.external_attr = 0o120777 << 16
        for member in [*names, link]:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); archive = root/'results.zip'
                make_archive(archive, extra=(member, 'payload'))
                with self.assertRaisesRegex(ValueError, 'Unsafe|symlinks'):
                    restore_previous_archive(archive, root/'local')
                self.assertFalse((root/'local').exists())

    def test_unreadable_or_corrupt_zip_fails_without_creating_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(OSError, 'cannot read the results ZIP'):
                restore_previous_archive(root/'missing.zip', root/'local')
            (root/'bad.zip').write_bytes(b'incomplete')
            with self.assertRaisesRegex(ValueError, 'incomplete or corrupt'):
                restore_previous_archive(root/'bad.zip', root/'local')
            self.assertFalse((root/'local').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
