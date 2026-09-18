"""Check installed AND already imported packages before loading model weights."""
from importlib.metadata import version
import sys


MODULE_NAMES = {'huggingface-hub': 'huggingface_hub'}


def runtime_versions(expected):
    report = {}
    for package, wanted in expected.items():
        module = sys.modules.get(MODULE_NAMES.get(package, package))
        loaded = getattr(module, '__version__', None) if module is not None else None
        report[package] = dict(expected=wanted, installed=version(package),
                               loaded=str(loaded) if loaded is not None else None,
                               imported=module is not None,
                               module_file=getattr(module, '__file__', None) if module is not None else None)
    return report


def verify_runtime(expected):
    report = runtime_versions(expected)
    installed = [f"{p}: expected {r['expected']}, installed {r['installed']}"
                 for p, r in report.items() if r['installed'] != r['expected']]
    stale = [f"{p}: installed {r['installed']}, loaded {r['loaded']}"
             for p, r in report.items() if r['loaded'] is not None and r['loaded'] != r['installed']]
    if installed or stale:
        details = '\n'.join(installed + stale)
        if stale:
            raise ValueError('Stale imported runtime; package installation did not replace modules in memory.\n'
                             + details + '\nRestart the notebook kernel, then rerun setup and imports. '
                             'No model has been loaded by this preflight.')
        raise ValueError('Runtime drift:\n' + details + '\nRestore all pinned packages, restart the kernel, '
                         'then rerun setup and imports.')
    return report
