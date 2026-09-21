"""Stage C prompt construction (SKILL §1b/§3 + §12 Task-1 outcome).

Measured policy (2026-08-10 smoke, human-approved): positives are
SUBJECT-PURE per channel; negatives only OUTSIDE neg_margin x the subject
extent (periphery rank alone poisoned small blobs); part-coverage
augmentation adds extremal-extent positives (rider-off-horse residual);
a small variant family is emitted and the gates referee selection.
Pure NumPy — no SAM2 here.
"""
from __future__ import annotations

import numpy as np


def _fps(pts, k, seed=0):
    if len(pts) <= k:
        return np.arange(len(pts))
    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[picked[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.asarray(picked)


def subject_anchor(sb, qc_b, cfg_c):
    """(channel, subject_idx, anchor_idx, fcr). Anchor = subject-pure set."""
    pc = cfg_c["prompts"]
    fcr = qc_b.get("follow_cam") or {}
    if fcr.get("branch_active"):
        return "branch", sb["fc_blob_idx"], sb["fc_blob_idx"], fcr
    osrc = sb["o_src_members"]
    anchor = np.array([], int)
    if fcr.get("fired") and "fc_blob_idx" in sb:
        blob = set(int(x) for x in sb["fc_blob_idx"])
        anchor = np.array([i for i in osrc if int(i) in blob], int)
    if len(anchor) < int(pc["n_pos_min"]):
        mm = sb["m"][osrc]
        anchor = osrc[mm >= np.nanquantile(mm, 0.75)]
    # third purity axis (measured 2026-08-10c): winner∩blob inherits the
    # fixation-shell/stray overlap (quiet AND mislabeled shore tracks) —
    # keep only the above-median-m half: quiet ∩ cluster ∩ MOVING excludes
    # shell members, whose m is marginal by construction
    if len(anchor) >= 2 * int(pc["n_pos_min"]):
        ma = sb["m"][anchor]
        keep = anchor[ma >= np.nanmedian(ma)]
        if len(keep) >= int(pc["n_pos_min"]):
            anchor = keep
    return "x0", osrc, anchor, fcr


def build_variants(b, sb, qc_b, cfg_c, eta=None):
    """Returns dict: t_c, subject core, prompt-variant family (process px),
    provenance record. Variants: V1_pts / V2_negs / V3_negs_box."""
    pc = cfg_c["prompts"]
    Q, P = b["Q"], b["P"]
    channel, subject, anchor, fcr = subject_anchor(sb, qc_b, cfg_c)
    um = np.nanmedian(np.where(Q, P[..., 0], np.nan), axis=1)
    vm = np.nanmedian(np.where(Q, P[..., 1], np.nan), axis=1)
    ctr = np.array([np.nanmedian(um[anchor]), np.nanmedian(vm[anchor])])
    d = np.hypot(um[subject] - ctr[0], vm[subject] - ctr[1])
    order = np.argsort(np.nan_to_num(d, nan=np.inf))
    n_core = max(int(len(subject) * float(pc["core_frac"])),
                 int(pc["n_pos_min"]))
    if channel == "x0":
        # §12 policy: X0 positives are the ANCHOR SET ITSELF (winner∩blob /
        # top-m) — the smoke's G_auxseed winner. Distance-ranked osrc core
        # re-diluted the horse's prompts with shore strays (measured Task-2
        # calibration regression, 2026-08-10: shore-ring mask, subject
        # excluded — the D_core failure reborn).
        core = anchor
    else:
        core = subject[order[:n_core]]
    peri = subject[order[-max(int(len(subject) * float(pc["periphery_frac"])), 1):]]

    if fcr.get("fired") and channel == "x0":
        t_c = int(fcr["seed_frame"])
    elif channel == "branch":
        t_c = int(fcr["seed_frame"])
    else:
        t_c = int(np.argmax(Q[core].sum(axis=0)))

    core_v = core[Q[core, t_c]]
    prompt_poor = len(core_v) < int(pc["n_pos_min"])
    cp = P[core_v, t_c].astype(np.float64)
    cp = cp[_fps(cp, int(pc["n_pos"]))]
    # part coverage: extremal-v (topmost) points from the ANCHOR-PURE core —
    # measured 2026-08-10b: drawing these from the full subject set injected
    # the topmost STRAYS (sky/trees, mural) as positives on both X0 clips;
    # the attached part (rider) sits directly above the core, so core-topmost
    # suffices and cannot contaminate
    subj_v = subject[Q[subject, t_c]]
    if len(core_v):
        topk = core_v[np.argsort(P[core_v, t_c, 1])[:max(int(pc["n_top"]), 0)]]
        cp = np.concatenate([cp, P[topk, t_c].astype(np.float64)])

    # scale-aware negatives (Task-1: 058's small blob put periphery next to
    # the subject): candidates = periphery + eta-reliable non-subject statics,
    # kept only beyond neg_margin x subject radius, nearest-beyond first
    ext = P[subj_v, t_c] if len(subj_v) else cp
    c_ext = ext.mean(axis=0)
    # p90, not max: one outlying subject member must not inflate the radius
    # past the frame and starve the negative pool
    r_s = float(np.quantile(np.linalg.norm(ext - c_ext, axis=1), 0.9)) \
        if len(ext) else 1.0
    cand = [i for i in peri if Q[i, t_c]]
    subj_set = set(int(x) for x in subject)
    stat = np.flatnonzero(~sb["y_dyn"] & Q[:, t_c])
    if eta is not None and len(stat):
        stat = stat[eta[stat, t_c] > float(np.nanmedian(eta[:, t_c]))]
    cand += [i for i in stat if int(i) not in subj_set]
    cand = np.array(cand, int)
    if len(cand):
        dc = np.linalg.norm(P[cand, t_c] - c_ext, axis=1)
        ok = cand[dc > float(pc["neg_margin"]) * r_s]
        dc = dc[dc > float(pc["neg_margin"]) * r_s]
        ok = ok[np.argsort(dc)]
        npn = P[ok[:int(pc["n_neg"]) * 3], t_c].astype(np.float64)
        npn = npn[_fps(npn, int(pc["n_neg"]), seed=1)]
    else:
        npn = np.zeros((0, 2))

    mrg = float(pc["box_margin_px"])
    tight = [cp[:, 0].min() - mrg, cp[:, 1].min() - mrg,
             cp[:, 0].max() + mrg, cp[:, 1].max() + mrg]
    box = ([float(v) for v in fcr["seed_bbox"]]
           if (channel == "x0" and fcr.get("fired")) else tight)
    lab = np.concatenate([np.ones(len(cp), np.int32),
                          np.zeros(len(npn), np.int32)])
    pn = np.concatenate([cp, npn])
    variants = {"V1_pts": (cp, np.ones(len(cp), np.int32), None),
                "V2_negs": (pn, lab, None),
                "V3_negs_box": (pn, lab, box)}
    prov = {"channel": channel, "t_c": t_c, "anchor_n": int(len(anchor)),
            "core_n": int(len(core)), "n_pos": int(len(cp)),
            "n_neg": int(len(npn)), "box": box, "r_s": round(r_s, 1),
            "prompt_poor": bool(prompt_poor),
            "points": {k: v[0].tolist() for k, v in variants.items()}}
    return t_c, core, variants, prov
