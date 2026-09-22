"""Stage C SAM2 adapter + gate referee (SKILL §4 + §12; Task-0 conventions
as runtime asserts: prompts = raw px at video-native res, logits at native
res binarized >0, bidirectional via reverse=True, in-place re-prompt)."""
from __future__ import annotations

import numpy as np

VARIANTS = {"tiny": "facebook/sam2.1-hiera-tiny",
            "small": "facebook/sam2.1-hiera-small",
            "base-plus": "facebook/sam2.1-hiera-base-plus",
            "large": "facebook/sam2.1-hiera-large"}


class Sam2Runner:
    def __init__(self, size, revision=None):
        """revision: pinned HF commit of the weights repo (ditflow extract-v1).
        Same config + checkpoint file as SAM2VideoPredictor.from_pretrained
        (which takes no revision), fetched at that commit."""
        import torch
        self.torch = torch
        if revision is None:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            self.predictor = SAM2VideoPredictor.from_pretrained(VARIANTS[size])
        else:
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2_video_predictor
            cfg_name, ckpt_name = HF_MODEL_ID_TO_FILENAMES[VARIANTS[size]]
            ckpt = hf_hub_download(VARIANTS[size], ckpt_name, revision=revision)
            self.predictor = build_sam2_video_predictor(cfg_name, ckpt)
        self.size = size
        self.revision = revision

    def run(self, frames_dir, n_fr, t_c, pts_native, labels, box_native):
        """seed + bidirectional propagation -> {t: bool mask (native res)}.
        Returns (masks, state) — state kept for QC re-prompting."""
        t = self.torch
        masks = {}
        with t.autocast("cuda", dtype=t.bfloat16), t.inference_mode():
            state = self.predictor.init_state(video_path=str(frames_dir))
            self.predictor.add_new_points_or_box(
                state, frame_idx=t_c, obj_id=1,
                points=pts_native.astype(np.float32), labels=labels,
                box=None if box_native is None
                else np.asarray(box_native, np.float32))
            for fi, _, ml in self.predictor.propagate_in_video(
                    state, start_frame_idx=t_c):
                masks[fi] = (ml[0, 0] > 0).cpu().numpy()
            for fi, _, ml in self.predictor.propagate_in_video(
                    state, start_frame_idx=t_c, reverse=True):
                masks[fi] = (ml[0, 0] > 0).cpu().numpy()
        assert len(masks) == n_fr, f"coverage {len(masks)}/{n_fr} (probe P1)"
        return masks, state

    def reprompt(self, state, masks, frame, pts_native, labels, span=12):
        """§4: inject at a failing frame (P3: bank updates in place), then
        re-propagate a bounded span forward."""
        t = self.torch
        with t.autocast("cuda", dtype=t.bfloat16), t.inference_mode():
            self.predictor.add_new_points_or_box(
                state, frame_idx=frame, obj_id=1,
                points=pts_native.astype(np.float32), labels=labels)
            for fi, _, ml in self.predictor.propagate_in_video(
                    state, start_frame_idx=frame, max_frame_num_to_track=span):
                masks[fi] = (ml[0, 0] > 0).cpu().numpy()
        return masks


def hull_iou(mask, pts):
    import cv2
    if len(pts) < 3:
        return float("nan")
    hull = cv2.convexHull(pts.astype(np.int32))
    canvas = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(canvas, hull, 1)
    inter = float(np.logical_and(mask, canvas).sum())
    union = float(np.logical_or(mask, canvas).sum())
    return inter / union if union else float("nan")


def score_masks(masks, core, P, Q, scale, n_fr):
    """Per-variant statistics for the gate referee."""
    ious, areas, hulls = [], [], []
    import cv2
    for t in range(n_fr):
        if t not in masks:
            continue
        mem = core[Q[core, t]]
        areas.append(float(masks[t].mean()))
        if len(mem) >= 3:
            p = P[mem, t] * scale
            ious.append(hull_iou(masks[t], p))
            h = cv2.convexHull(p.astype(np.int32))
            hulls.append(float(cv2.contourArea(h))
                         / (masks[t].shape[0] * masks[t].shape[1]))
    temp = [float(np.logical_and(masks[t], masks[t + 1]).sum()
                  / max(np.logical_or(masks[t], masks[t + 1]).sum(), 1))
            for t in range(n_fr - 1) if t in masks and t + 1 in masks]
    hull_med = float(np.nanmedian(hulls)) if hulls else float("nan")
    area_med = float(np.nanmedian(areas)) if areas else float("nan")
    return {"med_iou": float(np.nanmedian(ious)) if ious else float("nan"),
            "area_med": area_med, "hull_med": hull_med,
            "area_hull_ratio": area_med / hull_med if hull_med else float("nan"),
            "temp_med": float(np.nanmedian(temp)) if temp else float("nan")}


def referee(stats_by_variant, th):
    """§12.3: C-SIZE area band + C-ABS absolute ceiling screen FIRST (the
    hull-IoU referee alone crowned a 0.29-area anti-subject mask; the
    relative band alone passed whole-scene masks whose corrupted core hull
    was equally huge — corpus 2026-08-21), then hull-IoU ranks survivors;
    if nothing passes, fall back to best IoU with a flag."""
    lo, hi = float(th["c_size_lo"]), float(th["c_size_hi"])
    abs_max = float(th.get("c_abs_area_max", 1.0))
    inband = {k: s for k, s in stats_by_variant.items()
              if np.isfinite(s["area_hull_ratio"])
              and lo <= s["area_hull_ratio"] <= hi
              and s["area_med"] <= abs_max}
    pool = inband or stats_by_variant
    best = max(pool, key=lambda k: np.nan_to_num(pool[k]["med_iou"], nan=-1))
    return best, bool(inband)
