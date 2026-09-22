"""Config loader for Stage A (SKILL §1.4).

Single source of truth: modules take values from here, never from literals.
Supports ${a.b} interpolation, flags unfilled "<...>" placeholder fields, and
computes the content-addressing hash used by bundle caching (SKILL §5) so a
threshold change can never silently reuse a stale cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

_INTERP = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")
_PLACEHOLDER = re.compile(r"^<.*>$")

# Sections whose change must invalidate cached bundles.
# ditflow: "sam2" added so the SAM2 pins (stage_a.yaml) enter the content
# address — a model-version change must invalidate caches (extract-v1).
_HASHED_SECTIONS = ("d4rt", "sam2", "video", "thresholds")

# Content-addressing scope: ONLY the thresholds that shape Stage A bundle
# content enter Stage A's hash. Stage B gate values live in the same lock but
# cannot affect Stage A geometry — without this allowlist, adding Stage B
# entries spuriously invalidated every bundle (measured 2026-08-08). The
# filtered subset reproduces pre-Stage-B hashes exactly (same keys, values).
_STAGE_A_THRESHOLD_KEYS = (
    "tau_v", "tau_c", "tau_g_rel", "a_g2_min_inlier_frac", "tau_g_px",
    "tau_g2b_px", "a_cov_min_tracks", "a_cov_min_frac", "anchor_stride")


def _lookup(root: dict, dotted: str):
    node = root
    for part in dotted.split("."):
        node = node[part]
    return node


def _interpolate(node, root):
    if isinstance(node, dict):
        return {k: _interpolate(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root) for v in node]
    if isinstance(node, str):
        prev = None
        while prev != node:
            prev = node
            node = _INTERP.sub(lambda m: str(_lookup(root, m.group(1))), node)
        return node
    return node


def unfilled_fields(cfg: dict, prefix: str = "") -> list[str]:
    """Dotted paths of fields still holding '<...>' placeholders."""
    out: list[str] = []
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out += unfilled_fields(v, path + ".")
        elif isinstance(v, str) and _PLACEHOLDER.match(v.strip()):
            out.append(path)
    return out


def config_hash(cfg: dict) -> str:
    """sha256 (truncated) over the geometry/threshold-affecting subset."""
    subset = {k: cfg.get(k) for k in _HASHED_SECTIONS}
    th = subset.get("thresholds")
    if isinstance(th, dict):
        subset["thresholds"] = {k: th[k] for k in _STAGE_A_THRESHOLD_KEYS
                                if k in th}
    blob = json.dumps(subset, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path = "configs/stage_a.yaml",
                require_filled: tuple[str, ...] = ()) -> dict:
    """Load, interpolate, and validate the Stage A config.

    Gate thresholds are governed by configs/calibration.lock.yaml (READ-ONLY
    to Claude Code, CLAUDE.md §4): its `thresholds` section is merged over the
    stage config's, and its content participates in the config hash — so a
    human threshold change invalidates cached bundles exactly like any other
    geometry-affecting config change.

    require_filled: dotted prefixes that must not contain '<...>' placeholders
    for the calling step — e.g. the probe passes ("d4rt",); the full Stage A
    run passes ("repo", "d4rt"). Unrelated placeholders are tolerated but
    reported under cfg["_meta"]["unfilled"].
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = _interpolate(copy.deepcopy(raw), raw)

    lock_path = path.parent / "calibration.lock.yaml"
    if not lock_path.exists():
        raise FileNotFoundError(
            f"{lock_path} missing — gate thresholds live there (CLAUDE.md §4)")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("thresholds", {}).update(lock.get("thresholds", {}))

    missing = unfilled_fields(cfg)
    blocked = [m for m in missing if m.startswith(tuple(require_filled))] if require_filled else []
    if blocked:
        raise ValueError(
            f"{path} has unfilled fields required for this step: "
            + ", ".join(blocked)
            + "  — fill them once (human decision, SKILL §1.4) and re-run."
        )
    # Hash is computed on the MERGED config, so the lock file's semantic
    # content (not its comments) is part of the hash.
    cfg["_meta"] = {"config_hash": config_hash(cfg), "unfilled": missing,
                    "path": str(path), "lock_path": str(lock_path)}
    return cfg


def load_stage_a5(path_a5: str | Path = "configs/stage_a5.yaml",
                  path_a: str | Path = "configs/stage_a.yaml") -> dict:
    """Stage A.5 config: the Stage A config (with lock) plus the A.5 sections,
    with ${paths.*} in stage_a5.yaml interpolating against Stage A's paths.
    A.5 is read-only w.r.t. Stage A — its keys never enter the Stage A hash."""
    cfg = load_config(path_a)
    raw5 = yaml.safe_load(Path(path_a5).read_text(encoding="utf-8"))
    root = copy.deepcopy(raw5)
    root["paths"] = {**cfg["paths"], **raw5.get("paths", {})}
    a5 = _interpolate(copy.deepcopy(raw5), root)
    for key in ("vipe", "clips", "align"):
        cfg[key] = a5[key]
    cfg["paths"]["out_dir"] = a5["paths"]["out_dir"]
    return cfg
