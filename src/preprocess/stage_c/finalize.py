"""Stage C finalization (SKILL §7): the §1.5-complete bundle addendum.
M_src at PROCESS resolution (Task-0 P2: nearest-downsample, 0.1% area
drift), packed bits; absent fields absent, never placeholder-filled."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_stage_c(cfg, clip_id, masks_native, pres, t_c, prov, gate_stats,
                  reprompt_log, chosen, upstream_hash):
    import cv2
    Hp = int(cfg["video"]["process_height"])
    F = len(pres)
    M = np.zeros((F, Hp, Hp), bool)
    for t, m in masks_native.items():
        if 0 <= t < F and m.any():
            M[t] = cv2.resize(m.astype(np.uint8), (Hp, Hp),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    assert M.shape == (F, Hp, Hp)
    out = Path(cfg["paths"]["stage_c_cache"]) / clip_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "stage_c.npz",
                        M_src_packed=np.packbits(M),
                        M_shape=np.array(M.shape),
                        object_present=pres, t_c=np.array([t_c]))
    qc_c = {"clip_id": clip_id, "t_c": int(t_c), "chosen_variant": chosen,
            "prompts": prov, "gates": gate_stats,
            "reprompt_log": reprompt_log,
            "present_frames": int(pres.sum()),
            "config_hash": cfg["_meta"]["config_hash"],
            "upstream_bundle_hash": upstream_hash}
    (out / "qc_c.json").write_text(json.dumps(qc_c, indent=2),
                                   encoding="utf-8")
    return qc_c


def read_mask(path_dir):
    with np.load(Path(path_dir) / "stage_c.npz") as z:
        shape = tuple(z["M_shape"])
        M = np.unpackbits(z["M_src_packed"])[:int(np.prod(shape))]
        return M.reshape(shape).astype(bool), z["object_present"], int(z["t_c"][0])
