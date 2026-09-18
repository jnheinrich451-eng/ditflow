"""Catch pip downgrades that leave newer modules loaded, without any models."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.wan_runtime_preflight import verify_runtime


def main():
    expected = {'transformers': '4.57.6', 'huggingface-hub': '0.36.2'}
    modules = {'transformers': SimpleNamespace(__version__='5.16.1'),
               'huggingface_hub': SimpleNamespace(__version__='0.36.2')}
    with patch.dict(sys.modules, modules), patch('benchmark.wan_runtime_preflight.version', side_effect=expected.__getitem__):
        try:
            verify_runtime(expected)
        except ValueError as error:
            assert 'Stale imported runtime' in str(error)
            assert 'installed 4.57.6, loaded 5.16.1' in str(error)
            assert 'Restart' in str(error)
        else:
            raise AssertionError('Installed metadata hid a stale loaded Transformers')
        modules['transformers'].__version__ = '4.57.6'
        assert verify_runtime(expected)['transformers']['loaded'] == '4.57.6'
        modules['huggingface_hub'].__version__ = '1.0.0'
        try:
            verify_runtime(expected)
        except ValueError as error:
            assert 'huggingface-hub' in str(error) and 'loaded 1.0.0' in str(error)
        else:
            raise AssertionError('Hub module-name mapping failed')
    with patch.dict(sys.modules, {'transformers': None, 'huggingface_hub': None}), patch(
            'benchmark.wan_runtime_preflight.version', side_effect=expected.__getitem__):
        assert verify_runtime(expected)['transformers']['loaded'] is None
    with patch.dict(sys.modules, {'transformers': SimpleNamespace(__version__='5.16.1')}), patch(
            'benchmark.wan_runtime_preflight.version', return_value='5.16.1'):
        try:
            verify_runtime({'transformers': '4.57.6'})
        except ValueError as error:
            assert 'Runtime drift' in str(error)
        else:
            raise AssertionError('Actually wrong installed runtime accepted')
    print('PASS: stale in-memory Transformers/Hub, clean kernel, and installed-version drift checks; no model imports.')


if __name__ == '__main__':
    main()
