"""Frozen-contract bundle writer/reader + QC gates (SKILL §6, §7).

Layout: cache_dir/<clip_id>/{bundle.npz, provenance.json, qc.json}.
Staleness guard: config_hash is stored in both sidecars and validated by the
reader against the CURRENT config — a threshold change can never silently
reuse a stale cache (equivalent protection to hash-addressed paths, but
friendlier to the notebook handles).

A failed gate stops the run with the QC report; no bundle is emitted (§7).
Fields Stage A does not produce are absent, never placeholder-filled (§10.7).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# field -> (shape template, dtype); None dims are free but cross-checked below.
CONTRACT = {
    "K":       ((None, 3, 3), np.float32),
    "G":       ((None, 4, 4), np.float32),
    "D":       ((None, None, None), np.float16),
    "anchors": ((None, 3), np.float32),
    "X_local": ((None, None, 3), np.float32),
    "X0":      ((None, None, 3), np.float32),
    "P":       ((None, None, 2), np.float32),
    "V":       ((None, None), np.float32),
    "C":       ((None, None), np.float32),
    "Q":       ((None, None), np.bool_),
}


def _validate(arrays: dict) -> None:
    missing = set(CONTRACT) - set(arrays)
    assert not missing, f"contract fields missing: {sorted(missing)}"
    for name, (shape, dtype) in CONTRACT.items():
        a = arrays[name]
        assert a.dtype == dtype, f"{name}: dtype {a.dtype} != {np.dtype(dtype)}"
        assert a.ndim == len(shape), f"{name}: ndim {a.ndim} != {len(shape)}"
        for want, got in zip(shape, a.shape):
            assert want is None or want == got, f"{name}: shape {a.shape} vs {shape}"
        if a.dtype != np.bool_:
            assert np.isfinite(a).all(), f"{name}: non-finite values (A-FIN)"
    F = arrays["K"].shape[0]
    N = arrays["anchors"].shape[0]
    for name in ("G", "D"):
        assert arrays[name].shape[0] == F, f"{name}: frame dim != {F}"
    for name in ("X_local", "X0", "P", "V", "C", "Q"):
        assert arrays[name].shape[:2] == (N, F), f"{name}: shape[:2] != ({N},{F})"
    assert np.all(arrays["anchors"][:, 2] == np.round(arrays["anchors"][:, 2])), \
        "anchors: s_i must be int-valued"


def run_gates(tr: dict, cfg: dict) -> dict:
    """§7 gates on the assembled track dict (pre-write). Returns qc dict.

    A gate whose lock-file threshold is null reports its statistics with
    pass=None ("ungated — pending calibration", CLAUDE.md §4) and is excluded
    from all_pass; it never blocks a bundle and never silently passes one.
    """
    th = cfg["thresholds"]
    Q, V = tr["Q"], tr["V"]
    N, F = Q.shape
    gates = {}

    # A-G2 (Task 2 statistic change): gated on the clip-invariant RELATIVE 3D
    # residual + inlier fraction; the pixel median is report/viz-only.
    rel = tr["eq24_rel"][Q]
    rel = rel[np.isfinite(rel)]
    med_rel = float(np.median(rel)) if rel.size else float("nan")
    # inlier = share of valid points with relative residual within 5x the clip
    # median (the 5x is part of the statistic's definition, Task 2). <= not <:
    # equivalent for continuous data, correct when residuals are exactly zero.
    inlier = float((rel <= 5.0 * med_rel).mean()) if rel.size else float("nan")
    px = tr["eq24_px"][Q]
    px = px[np.isfinite(px)]
    tau_rel = th.get("tau_g_rel")
    min_inl = th.get("a_g2_min_inlier_frac")
    if tau_rel is None or min_inl is None:
        g2_pass = None  # ungated — pending calibration (CLAUDE.md §4)
    else:
        g2_pass = bool(med_rel < float(tau_rel) and inlier >= float(min_inl))
    gates["A-G2"] = {"median_rel": med_rel, "inlier_frac": inlier,
                     "median_px_viz_only": float(np.median(px)) if px.size else float("nan"),
                     "tau_rel": tau_rel, "min_inlier_frac": min_inl,
                     "pass": g2_pass}

    s = tr["anchors"][:, 2].astype(int)
    p_at_s = tr["P"][np.arange(N), s]
    g2b = float(np.median(np.linalg.norm(p_at_s - tr["anchors"][:, :2], axis=1)))
    gates["A-G2b"] = {"median_px": g2b, "tau": th["tau_g2b_px"],
                      "pass": bool(g2b < th["tau_g2b_px"])}

    bad_ch = float((tr["X_local"][..., 2][Q] <= 0).mean()) if Q.any() else 0.0
    gates["A-CH"] = {"frac_z_nonpos": bad_ch, "pass": bool(bad_ch == 0.0)}

    min_tracks = int(th["a_cov_min_tracks"])
    cov = float((Q.sum(axis=0) >= min_tracks).mean())
    gates["A-COV"] = {"frac_frames_ok": cov, "min_tracks": min_tracks,
                      "min_frac": th["a_cov_min_frac"],
                      "pass": bool(cov > float(th["a_cov_min_frac"]))}

    ungated = sorted(k for k, g in gates.items() if g["pass"] is None)
    return {"gates": gates,
            "all_pass": all(g["pass"] for g in gates.values() if g["pass"] is not None),
            "ungated_pending_calibration": ungated,
            "n_anchors": int(N), "n_frames": int(F),
            "sim3_scale_minmax": [float(tr["sim3_scale"].min()),
                                  float(tr["sim3_scale"].max())],
            # full per-frame scale curve: Stage B's b_t drift correction reads
            # this (signed-off 2026-08-01: drift recorded, not gated, in Stage A)
            "sim3_scale_per_frame": [float(v) for v in tr["sim3_scale"]],
            "thresholds_used": {k: th.get(k) for k in
                                ("tau_v", "tau_c", "tau_g_rel",
                                 "a_g2_min_inlier_frac", "tau_g_px",
                                 "tau_g2b_px", "a_cov_min_tracks",
                                 "a_cov_min_frac", "anchor_stride")}}


def write_bundle(cfg: dict, clip_id: str, arrays: dict,
                 provenance: dict, qc: dict) -> Path:
    _validate(arrays)
    # A-G8: provenance present, single geometry source, config hash recorded.
    for key in ("geometry_source", "model_commit", "checkpoint", "gpu", "env"):
        assert key in provenance, f"provenance missing '{key}' (A-G8)"
    out = Path(cfg["paths"]["cache_dir"]) / clip_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "bundle.npz", **arrays)
    stamp = {"config_hash": cfg["_meta"]["config_hash"], "clip_id": clip_id}
    (out / "provenance.json").write_text(
        json.dumps({**provenance, **stamp}, indent=2), encoding="utf-8")
    (out / "qc.json").write_text(
        json.dumps({**qc, **stamp}, indent=2), encoding="utf-8")
    return out


def read_bundle(cfg: dict, clip_id: str, allow_stale: bool = False) -> dict:
    """Load + re-validate a bundle; refuses malformed or stale-config bundles.

    allow_stale=True is for READ-ONLY diagnostics on bundles whose config
    hash predates a non-geometric change (caller must warn loudly); pipeline
    consumers never set it."""
    d = Path(cfg["paths"]["cache_dir"]) / clip_id
    for f in ("bundle.npz", "provenance.json", "qc.json"):
        assert (d / f).exists(), f"malformed bundle: {d / f} missing"
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    if prov.get("config_hash") != cfg["_meta"]["config_hash"] and not allow_stale:
        raise RuntimeError(
            f"stale bundle {d.name}: built with config {prov.get('config_hash')}, "
            f"current is {cfg['_meta']['config_hash']} — re-run Stage A")
    arrays = dict(np.load(d / "bundle.npz"))
    _validate(arrays)
    arrays["provenance"] = prov
    arrays["qc"] = json.loads((d / "qc.json").read_text(encoding="utf-8"))
    return arrays
