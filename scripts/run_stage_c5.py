#!/usr/bin/env python
"""Stage C.5 runner (SKILL v1.1): mask-guided dense D4RT tracks, one pass.
GPU (re-encodes per latent_cache_policy). Channel-scoped gates per §1b.

    python scripts/run_stage_c5.py --clip-id <id> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                            # noqa: E402
from src.preprocess.bundle import read_bundle          # noqa: E402
from src.preprocess.config import load_config          # noqa: E402
from src.preprocess.stage_c import finalize            # noqa: E402
from src.preprocess.stage_c5 import densify            # noqa: E402

REQ = ("X_local", "X0", "V", "C")


def project(Xl, K):
    """P from X_local via per-frame K (pinhole) — same derivation as Stage A;
    invalid-z points produce garbage uv that the Q gate removes."""
    Xl = np.asarray(Xl, np.float64)
    z = np.where(np.abs(Xl[..., 2]) > 1e-9, Xl[..., 2], 1e-9)
    u = K[None, :, 0, 0] * Xl[..., 0] / z + K[None, :, 0, 2]
    v = K[None, :, 1, 1] * Xl[..., 1] / z + K[None, :, 1, 2]
    return np.stack([u, v], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", nargs="*", default=[])
    ap.add_argument("--curated", default=None,
                    help="tier from configs/curated_clips.yaml "
                         "(tier_a_good | tier_b_part_missing | all)")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--config-c5", default="configs/stage_c5.yaml")
    args = ap.parse_args()
    if args.curated:
        cur = yaml.safe_load(Path("configs/curated_clips.yaml")
                             .read_text(encoding="utf-8"))
        tiers = list(cur) if args.curated == "all" else [args.curated]
        args.clip_id = [c for t in tiers for c in cur[t]]
    assert args.clip_id, "give --clip-id ... or --curated <tier>"
    cfg = load_config(args.config)
    c5 = yaml.safe_load(Path(args.config_c5).read_text(encoding="utf-8"))["c5"]
    out_root = Path(str(cfg["paths"]["drive_root"]) + "/cache/stage_c5")
    out_root.mkdir(parents=True, exist_ok=True)
    th = cfg["thresholds"]

    from src.preprocess.d4rt_adapter import D4RTAdapter
    import imageio.v2 as iio
    import cv2
    adapter = D4RTAdapter(cfg)
    rows = []
    for clip_id in args.clip_id:
        b = read_bundle(cfg, clip_id)
        F = b["Q"].shape[1]
        Hp = int(cfg["video"]["process_height"])
        M, present, t_c = finalize.read_mask(
            Path(cfg["paths"]["stage_c_cache"]) / clip_id)
        qc_b = json.loads((Path(cfg["paths"]["stage_b_cache"])
                           / f"{clip_id}_qc_b.json").read_text(encoding="utf-8"))
        channel = "branch" if (qc_b.get("follow_cam") or {}).get(
            "branch_active") else "x0"
        rd = iio.get_reader(b["provenance"]["clip_path"])
        frames = [cv2.resize(fr[..., :3], (Hp, Hp),
                             interpolation=cv2.INTER_AREA)
                  for i, fr in enumerate(rd) if i < F]
        rd.close()
        adapter.open(np.stack(frames), b["provenance"]["clip_path"])

        stride = int(c5["stride_dense"])
        if int(M.sum(axis=(1, 2)).max()) < int(c5["small_subject_px"]):
            stride = 1
        occ = np.zeros((Hp, Hp), bool)
        parts, anchors = [], []
        for t in range(F):
            if not present[t]:
                continue
            # dedup vs Stage A tracks live at t
            occ_t = occ.copy()
            live = b["Q"][:, t]
            densify.mark(occ_t, b["P"][live, t], stride / 2)
            mt = densify.dilate_mask(M[t], int(c5["dilate_px"]))
            uv = densify.candidates(mt, stride, occ_t)
            if not len(uv):
                continue
            if sum(len(p["P"]) for p in parts) + len(uv) > int(c5["n_dense_max"]):
                uv = uv[:max(int(c5["n_dense_max"])
                             - sum(len(p["P"]) for p in parts), 0)]
            if not len(uv):
                break
            r = adapter.track_batch(uv, t)
            missing = [k for k in REQ if k not in r]
            assert not missing, f"track_batch lacks {missing} — keys: {list(r)}"
            part = {k: np.asarray(r[k]) for k in REQ}
            part["P"] = project(part["X_local"], b["K"].astype(np.float64))
            parts.append(part)
            anchors.append(np.column_stack([uv, np.full(len(uv), t)]))
            for tt in range(F):   # local occupancy: seeded tracks claim pixels
                densify.mark(occ, part["P"][:, tt], stride / 2)
        assert parts, f"{clip_id}: zero dense anchors — seeding/mask bug"
        A = {k: np.concatenate([p[k] for p in parts]) for k in REQ + ("P",)}
        anchors = np.concatenate(anchors)
        N = len(anchors)

        z_ok = A["X_local"][..., 2] > 0
        u01 = (A["P"][..., 0] >= 0) & (A["P"][..., 0] < Hp) \
            & (A["P"][..., 1] >= 0) & (A["P"][..., 1] < Hp)
        Q = (A["V"] >= float(th["tau_v"])) & (A["C"] >= float(th["tau_c"])) \
            & z_ok & u01
        eta, sig_d, e = densify.eta_of(A["X0"], A["X_local"], b["G"], Q)
        Z = np.maximum(A["X_local"][..., 2], 0.05 * np.median(
            A["X_local"][..., 2][Q]))
        g2 = float(np.median((e / Z)[Q]))
        s_i = anchors[:, 2].astype(int)
        repro = float(np.median(np.linalg.norm(
            A["P"][np.arange(N), s_i] - anchors[:, :2], axis=1)))
        pure = densify.inside_mask_frac(A["P"], Q, M)
        gates = {"channel": channel,
                 "C5-G2b": {"px": round(repro, 2),
                            "pass": bool(repro <= float(th["tau_g2b_px"]))},
                 "C5-G2": {"rel": round(g2, 4),
                           "pass": bool(g2 <= float(th["tau_g_rel"]))
                           if channel == "x0" else None},
                 "C5-COUNT": {"n": int(N), "pass": bool(N <= int(c5["n_dense_max"]))},
                 "C5-PURITY": {"med": round(float(np.nanmedian(pure)), 3),
                               "pass": None if th.get("tau_pure") is None
                               else bool(np.nanmedian(pure) >= float(th["tau_pure"]))}}
        np.savez_compressed(out_root / f"{clip_id}.npz",
                            anchors_dense=anchors.astype(np.float32),
                            X_local_dense=A["X_local"].astype(np.float32),
                            X0_dense=A["X0"].astype(np.float32),
                            P_dense=A["P"].astype(np.float32),
                            Q_dense=Q, eta_dense=eta.astype(np.float32),
                            inside_mask_frac=pure.astype(np.float32))
        (out_root / f"{clip_id}_qc_c5.json").write_text(json.dumps(
            {"clip_id": clip_id, "channel": channel, "n_dense": int(N),
             "sigma_eta_dense": sig_d, "gates": gates,
             "config_hash": cfg["_meta"]["config_hash"],
             "upstream": {"bundle": b["provenance"].get("config_hash")}},
            indent=2), encoding="utf-8")
        rows.append(f"{clip_id}: {channel} n={N} G2b={repro:.2f}px "
                    f"G2={g2:.4f} purity={np.nanmedian(pure):.3f}")
        print(rows[-1])
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
