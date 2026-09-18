"""Verify parity-source reuse, including audited legacy mixed line endings.

The legacy bundle hashed raw files with mixed CRLF/LF endings. Git's LF blobs
cannot reproduce those raw hashes by converting every line to LF or CRLF.
The four mappings below were audited against the retained original bytes:
each raw hash matches the saved acceptance manifest, and replacing only CRLF
with LF exactly matches Git commit f1a0b5322311aeb5148daf1024dce872e253e19d.
No whitespace, comments, or executable code are otherwise ignored.
"""
import hashlib


CORE_SOURCES = (
    'motion_guidance_wan.py', 'guidance_utils/wan_transformer.py',
    'guidance_utils/wan_modules.py', 'guidance_utils/wan_motion_flow_utils.py',
    'guidance_utils/wan_guidance_schedule.py', 'guidance_utils/motion_probe.py',
    'configs/guidance_config_wan.yaml',
)

# (file, verified saved raw SHA256) -> verified LF-normalized SHA256.
AUDITED_MIXED_EOL = {
    ('motion_guidance_wan.py',
     '13c45173149c661e427bbcc4677b49c3b6bae145b0a7bcb0ef3ce237dca58c16'):
        '7d8780a69ce31368bdbfefc5d8ec924267ecdc4b10cd02b919c80b5ecc89418a',
    ('guidance_utils/wan_modules.py',
     'e8e78295a9f6ec7f1ce4855e487ea7b679902dc54a66769ffe79d2dd8cb6614e'):
        'a6cf6ed885f1ecfe256bb0fffb3ffddc4533774925e97a85977d888483955e85',
    ('guidance_utils/wan_motion_flow_utils.py',
     '5e360a16bf7dcac4dc1ca8267bef604907f0db8b2edf46dac66d43a16eceec28'):
        '20dfb16660b11937d0c216e036e3f9f8ff5dd8db62f72e9d3ac68a74f9e586b4',
    ('configs/guidance_config_wan.yaml',
     '9374a6538758bbf9d5cb14b61a8bf98c45f46aa71ca427f9e068715875fecfa4'):
        'af84fa6618ea846c6e65907f8cc8d001fd6faaded778736fc318fe9a06c8a920',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(name, raw, expected):
    lf = raw.replace(b'\r\n', b'\n')
    actual, normalized = digest(raw), digest(lf)
    if actual == expected:
        method = 'exact'
    elif expected in (normalized, digest(lf.replace(b'\n', b'\r\n'))):
        method = 'uniform_line_endings'
    elif AUDITED_MIXED_EOL.get((name, expected)) == normalized:
        method = 'audited_mixed_line_endings'
    else:
        raise ValueError(
            f'Core source changed; cannot reuse native parity: {name}\n'
            f'Saved SHA256={expected}; current SHA256={actual}; LF SHA256={normalized}. '
            'No verified line-ending-only match. Preserve this report and inspect the source difference.'
        )
    return dict(file=name, method=method, saved_sha256=expected,
                current_sha256=actual, lf_sha256=normalized)


def verify_parity_sources(root, recorded):
    return [verify_source(name, (root / name).read_bytes(), recorded[name]) for name in CORE_SOURCES]
