"""D4RT adapter — the ONLY module that knows model conventions (SKILL §5).

Downstream modules consume the frozen convention regardless of what the model
natively returns: OpenCV camera axes, G_{0<-t} (camera-t -> camera-0), z-depth,
process-resolution pixels, V/C in [0,1].

Probe-derived facts encoded here (probe_report 2026-08-01, human-reviewed):
  P1  heads: xyz_3d, uv_2d (normalized), visibility (LOGIT), confidence.
  P2  derived G maps camera-t coords -> camera-0 coords (margin 3.4x).
  P3  z-depth semantics (vs ray-length: 1.1% vs 12.9% rel err).
  P4  indices 0-based; model silently ACCEPTS out-of-range indices, so this
      adapter range-asserts every call itself.
  P5  K source = least-squares pinhole fit (beats the repo's median-fx/fy
      estimator); model self-consistency floor ~2 px @256^2 -> tau_g2b_px.
  P7  latent_cache_policy: reencode (encode 0.4 s, F = 0.044 GB).
  --  OpenD4RT has NO camera heads: K and G are DERIVED from grid queries per
      the repo's own geometry_decoding spec (rigid umeyama; per-frame sim3
      scale kept as a drift diagnostic).
  --  temporal window = clip_frames (48); longer clips are handled by
      per-target windows that ALWAYS contain global frame 0 (and the track's
      source frame), so t_cam=0 stays global camera-0 in every window.

IMPORT ORDER WARNING: Open-d4rt's package is also named `src`. Instantiating
D4RTAdapter evicts our `src` from sys.modules — import every needed
src.preprocess.* module BEFORE constructing the adapter (run_stage_a.py does).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_D4RT = {}  # populated once by _import_d4rt


def canonical_block_start(t: int, T: int, clip_frames: int) -> int:
    """Start frame of the canonical (source-independent) encode block for
    target t — the ONE source of truth for the window schedule, kept
    model-free so qc tooling can use it without instantiating the adapter."""
    span = clip_frames - 1                    # frame 0 takes the other slot
    step = max(1, clip_frames // 4)
    b = ((t - span // 2) // step) * step
    return max(1, min(b, T - span))


def window_boundaries(T: int, clip_frames: int) -> list[int]:
    """Frames whose canonical encode window differs from the previous
    frame's — the scale re-gauge points Stage B's drift model needs
    (Task 0 of the Stage B SKILL). Empty for single-window clips."""
    if T <= clip_frames:
        return []
    return [t for t in range(1, T)
            if canonical_block_start(t, T, clip_frames)
            != canonical_block_start(t - 1, T, clip_frames)]


def _import_d4rt(d4rt_dir: str) -> dict:
    """Import Open-d4rt's `src` package (one-time eviction of ours)."""
    if _D4RT:
        return _D4RT
    for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    while d4rt_dir in sys.path:
        sys.path.remove(d4rt_dir)
    sys.path.insert(0, d4rt_dir)
    from infer_track_3d import _resize_video, _unwrap_state_dict  # noqa: E402
    from src.core import load_checkpoint, load_yaml_config  # noqa: E402
    from src.eval.tasks import (  # noqa: E402
        _encode_model_memory, _model_clip_frames, _run_model_for_queries,
        _umeyama_rigid, _umeyama_sim3)
    from src.model import build_model  # noqa: E402
    _D4RT.update(resize_video=_resize_video, unwrap=_unwrap_state_dict,
                 load_checkpoint=load_checkpoint, load_yaml_config=load_yaml_config,
                 encode_memory=_encode_model_memory, clip_frames=_model_clip_frames,
                 run_queries=_run_model_for_queries, umeyama_rigid=_umeyama_rigid,
                 umeyama_sim3=_umeyama_sim3, build_model=build_model)
    return _D4RT


class D4RTAdapter:
    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tau_v = float(cfg["thresholds"]["tau_v"])
        m = _import_d4rt(cfg["paths"]["d4rt_dir"])

        ckpt_root = Path(cfg["paths"]["ckpt_dir"])
        sub = cfg["d4rt"]["checkpoint_subdir"]     # pinned name, never a name heuristic
        cands = sorted(p for p in ckpt_root.rglob("opend4rt.ckpt") if p.parent.name == sub)
        if len(cands) != 1:
            raise FileNotFoundError(
                f"expected exactly one {sub}/opend4rt.ckpt under {ckpt_root}, found "
                f"{[str(p) for p in cands]} — run scripts/bootstrap_extract.sh")
        ckpt = cands[0]
        model_cfg = m["load_yaml_config"](str(ckpt.parent / "model.yaml"))
        model = m["build_model"](model_cfg["model"]).eval()
        state = m["unwrap"](m["load_checkpoint"](ckpt, map_location="cpu"))
        if not state:
            raise RuntimeError(f"no weights inside {ckpt}")
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device).eval()

        image_size = model_cfg.get_path("model.input.image_size", [256, 256])
        assert int(image_size[0]) == int(cfg["video"]["process_height"]), \
            f"config video.process_height={cfg['video']['process_height']} but model wants {image_size[0]}"
        self.process_hw = (int(image_size[0]), int(image_size[1]))
        self.clip_frames = int(m["clip_frames"](model))
        self.checkpoint_name = ckpt.parent.name
        self._m = m
        self._uv_scale = np.asarray([max(self.process_hw[1] - 1, 1),
                                     max(self.process_hw[0] - 1, 1)], np.float64)

    # ---------------------------------------------------------------- session
    def open(self, video_u8: np.ndarray, clip_path: str = "") -> None:
        """Register the clip (native u8 [T,H,W,3]); resizes once to process res."""
        assert video_u8.ndim == 4 and video_u8.dtype == np.uint8
        T = int(video_u8.shape[0])
        assert T <= int(self.cfg["video"]["max_frames"]), \
            f"{T} frames > video.max_frames={self.cfg['video']['max_frames']}"
        self.T = T
        self.native_hw = tuple(int(v) for v in video_u8.shape[1:3])
        Hm, Wm = self.process_hw
        self.video_model = self._m["resize_video"](video_u8, (Hm, Wm))
        self.clip_path = clip_path
        self._stitch_cache = {}
        self._ref_window = None
        # §5 intrinsics rule: the resize is anisotropic; factors go to provenance.
        self.resize_factors = {"sx": Wm / self.native_hw[1], "sy": Hm / self.native_hw[0]}
        # LRU of encoder memory tokens per window, parked on CPU (~35 MB each).
        # Encoding is the runtime bottleneck for stitched clips; distinct windows
        # number ~T-clip_frames+1, so most encodes become cache hits.
        self._enc_cache: dict = {}
        self._enc_cap = 60
        self._canon: dict = {}  # per-target canonical windows (source-independent)
        # Window-stitch corrections (sim3 per window -> reference window frame).
        self._stitch_cache: dict = {}
        self._ref_window = None
        self._K = None

    # ---------------------------------------------------------------- windows
    def _window(self, t: int, s: int) -> tuple:
        """Frame window for target t of a track sourced at s: always contains
        global frame 0 (keeps t_cam=0 = global camera-0), s and t, and is
        always FULL SIZE (min(T, clip_frames)) — the encoder's temporal patch
        kernel (2 frames) cannot fit undersized windows.

        Windows are QUANTIZED: the contiguous block around t snaps to a coarse
        grid (clip_frames/4 steps), so an 81-frame clip uses ~4 canonical
        windows instead of one per target — that is what makes the encode LRU
        effective (every one-frame difference forces a full re-encode). A
        source outside the block swaps in for the block's last frame, which
        also dedupes across targets sharing the block."""
        T, clip = self.T, self.clip_frames
        if T <= clip:
            return tuple(range(T))
        t, s = int(t), int(s)
        win = self._canon.get(t)
        if win is None:
            b = canonical_block_start(t, T, clip)
            win = (0,) + tuple(range(b, b + clip - 1))
            assert t in win, "window quantization broke target membership"
            self._canon[t] = win
        if s in win:
            return win
        drop = win[-1] if win[-1] != t else win[-2]  # deterministic, t-independent
        return tuple(sorted((set(win) - {drop}) | {s}))

    def _encode(self, window: tuple) -> dict:
        torch, m = self.torch, self._m
        Hm, Wm = self.process_hw
        # The video tensor is rebuilt per call (cheap slice+upload); only the
        # encoder memory — the expensive part — is cached.
        vid = (torch.from_numpy(self.video_model[list(window)])
               .to(device=self.device, dtype=torch.float32)
               .permute(0, 3, 1, 2).unsqueeze(0) / 255.0)
        aspect = torch.tensor([[float(Wm) / float(Hm)]], dtype=torch.float32,
                              device=self.device)
        cached = self._enc_cache.pop(window, None)
        if cached is not None:
            self._enc_cache[window] = cached  # re-insert = mark most recent
            memory = cached.to(self.device) if cached is not None else None
        else:
            with torch.no_grad():
                memory = m["encode_memory"](model=self.model, video_b=vid,
                                            aspect_b=aspect)
            if memory is not None:
                self._enc_cache[window] = memory.detach().cpu()
                while len(self._enc_cache) > self._enc_cap:
                    self._enc_cache.pop(next(iter(self._enc_cache)))  # evict LRU
        return {"video": vid, "memory": memory, "aspect": aspect}

    # ---------------------------------------------------------------- queries
    def _query_local_idx(self, F, uv, ts, tt, tc) -> dict:
        """Raw query with window-LOCAL indices; returns canonical numpy dict."""
        torch, m = self.torch, self._m
        n_win = F["video"].shape[1]
        for arr in (ts, tt, tc):  # P4: the model accepts garbage indices silently
            assert arr.min() >= 0 and arr.max() < n_win, "window-local index out of range"
        uvn = np.asarray(uv, np.float64) / self._uv_scale
        assert uvn.min() >= 0 and uvn.max() <= 1.0 + 1e-6, "uv outside process resolution"
        mkf = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(
            device=self.device, dtype=torch.float32)
        mki = lambda a: torch.from_numpy(np.asarray(a, np.int64)).to(self.device)
        q = {"u": mkf(uvn[:, 0]), "v": mkf(uvn[:, 1]),
             "t_src": mki(ts), "t_tgt": mki(tt), "t_cam": mki(tc)}
        with torch.no_grad():
            pred = m["run_queries"](model=self.model, video_b=F["video"],
                                    aspect_b=F["aspect"], query=q,
                                    chunk_size=8192, memory_b=F["memory"])
        X = pred["xyz_3d"].numpy().astype(np.float32)
        # Both heads are trained as sigmoid quantities (src/losses/d4rt_loss.py:
        # visibility = BCE on logits; confidence c = sigmoid(raw) weighting the
        # xyz error). Raw confidence sits ~[6.6, 10.1] on this checkpoint, so
        # sigmoid(C) saturates near 1 — informative only in ranking, and tau_c
        # is effectively permissive.
        V = (1.0 / (1.0 + np.exp(-pred["visibility"].numpy()))).astype(np.float32)
        C = (1.0 / (1.0 + np.exp(-pred["confidence"].numpy()))).astype(np.float32)
        for name, a in (("X", X), ("V", V), ("C", C)):
            assert np.isfinite(a).all(), f"non-finite {name} from model"
        return {"X": X, "V": V, "C": C}

    # ------------------------------------------------------------- stitching
    # MEASURED (x0_static_drift, 2026-08-02): the model's t_cam=0 answers shift
    # when the encode window changes — every window builds its own notion of
    # the camera-0 frame, producing step discontinuities in X0/G at window
    # switches. Correction (human-signed 2026-08-02): per window, a sim3 fitted
    # on shared points queried at overlap frames in BOTH windows maps its
    # camera-0 coords into the reference window's frame — the same overlap
    # alignment Open-d4rt's own umeyama_slide_window inference mode uses. The
    # Eq 22 query path stays primary; this only reconciles window frames.

    def _stitch(self, window: tuple):
        """(scale, R, t) mapping this window's cam-0 coords -> reference
        window's cam-0 frame; (None, None, None) means identity."""
        if self._ref_window is None:
            self._ref_window = self._window(0, 0)
        if window == self._ref_window:
            return (None, None, None)
        got = self._stitch_cache.get(window)
        if got is not None:
            return got
        ref = self._ref_window
        Hm, Wm = self.process_hw
        gs = np.linspace(8, min(Hm, Wm) - 8, 16, dtype=np.float32)
        uv = np.stack(np.meshgrid(gs, gs), -1).reshape(-1, 2)
        ov = sorted(set(window) & set(ref))
        picks = sorted({ov[len(ov) // 4], ov[len(ov) // 2], ov[(3 * len(ov)) // 4]})
        A, B = [], []
        for t in picks:
            for win, acc in ((window, A), (ref, B)):
                F = self._encode(win)
                li = {f: i for i, f in enumerate(win)}
                idx = np.full(len(uv), li[t], np.int64)
                r = self._query_local_idx(F, uv, idx, idx,
                                          np.full(len(uv), li[0], np.int64))
                acc.append(r["X"])
        A, B = np.concatenate(A), np.concatenate(B)
        ok = (A[:, 2] > 0) & (B[:, 2] > 0)
        got = (None, None, None)
        if ok.sum() >= 8:
            sim = self._m["umeyama_sim3"](A[ok].astype(np.float64),
                                          B[ok].astype(np.float64))
            if sim is not None:
                s, R, tvec = sim
                got = (float(s), R.astype(np.float32), tvec.astype(np.float32))
        self._stitch_cache[window] = got
        return got

    def _apply_stitch(self, window: tuple, X: np.ndarray) -> np.ndarray:
        if self.T <= self.clip_frames:
            return X  # single window — nothing to reconcile
        s, R, t = self._stitch(window)
        if s is None:
            return X
        return (s * (X @ R.T) + t).astype(np.float32)

    def query_global(self, uv: np.ndarray, s: int, t: int, t_cam: int) -> dict:
        """Query anchors (process-res px, sourced at global frame s) at global
        target frame t in camera t_cam (t_cam must be t or 0). Camera-0
        results are window-stitch corrected (see _stitch)."""
        assert t_cam in (t, 0), "Stage A only uses local (t) and shared (0) cameras"
        window = self._window(t, s)
        li = {f: i for i, f in enumerate(window)}
        F = self._encode(window)
        n = len(uv)
        mk = lambda idx: np.full(n, li[idx], np.int64)
        out = self._query_local_idx(F, uv, mk(s), mk(t), mk(t_cam))
        if t_cam == 0:
            out = {**out, "X": self._apply_stitch(window, out["X"])}
        return out

    def track_batch(self, uv: np.ndarray, s: int) -> dict:
        """Full-clip local + shared trajectories for anchors spawned at s.
        Shared coords are RE-QUERIED with t_cam=0 (Eq 22 path — never obtained
        by transforming X_local; that transform is only the QC check).

        Target frames sharing a window are batched into ONE query call (all of
        them, for clips within clip_frames) — the model chunks internally, so
        big batches cost decode time only, not per-call overhead."""
        n, T = len(uv), self.T
        Xl = np.zeros((n, T, 3), np.float32)
        X0 = np.zeros((n, T, 3), np.float32)
        V = np.zeros((n, T), np.float32)
        C = np.zeros((n, T), np.float32)
        groups: dict[tuple, list[int]] = {}
        for t in range(T):
            groups.setdefault(self._window(t, int(s)), []).append(t)
        for window, ts in groups.items():
            F = self._encode(window)
            li = {f: i for i, f in enumerate(window)}
            k = len(ts)
            uv_rep = np.repeat(np.asarray(uv, np.float32), k, axis=0)  # anchor-major
            tt = np.tile(np.asarray([li[t] for t in ts], np.int64), n)
            ss = np.full(n * k, li[int(s)], np.int64)
            loc = self._query_local_idx(F, uv_rep, ss, tt, tt)
            ref = self._query_local_idx(F, uv_rep, ss, tt, np.full(n * k, li[0], np.int64))
            cols = np.asarray(ts)
            Xl[:, cols] = loc["X"].reshape(n, k, 3)
            X0[:, cols] = self._apply_stitch(window, ref["X"]).reshape(n, k, 3)
            V[:, cols] = loc["V"].reshape(n, k)
            C[:, cols] = loc["C"].reshape(n, k)
        return {"X_local": Xl, "X0": X0, "V": V, "C": C}

    # ---------------------------------------------------------------- cameras
    def cameras(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-frame K (LSQ pinhole fit, P5) and G_{0<-t} (rigid umeyama, P2
        direction) from a 32x32 same-frame query grid. Returns (K, G, sim3)."""
        if self._K is not None:
            return self._K, self._G, self._sim3
        m = self._m
        T = self.T
        Hm, Wm = self.process_hw
        gs = 32
        us = np.linspace(0, Wm - 1, gs, dtype=np.float32)
        vs = np.linspace(0, Hm - 1, gs, dtype=np.float32)
        uv = np.stack(np.meshgrid(us, vs), -1).reshape(-1, 2)
        K = np.zeros((T, 3, 3), np.float32)
        G = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        sim3 = np.ones(T, np.float32)
        for t in range(T):
            loc = self.query_global(uv, t, t, t)
            ref = self.query_global(uv, t, t, 0)
            ok = (loc["V"] > self.tau_v) & (loc["X"][:, 2] > 0)
            if ok.sum() < 16:
                ok = np.ones(len(uv), bool)
            X, uvq = loc["X"][ok], uv[ok]
            xz, yz = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
            (fx, cx), _, _, _ = np.linalg.lstsq(
                np.stack([xz, np.ones_like(xz)], 1), uvq[:, 0], rcond=None)
            (fy, cy), _, _, _ = np.linalg.lstsq(
                np.stack([yz, np.ones_like(yz)], 1), uvq[:, 1], rcond=None)
            K[t] = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
            rig = m["umeyama_rigid"](X.astype(np.float64), ref["X"][ok].astype(np.float64))
            if rig is not None:
                G[t, :3, :3], G[t, :3, 3] = rig
            sim = m["umeyama_sim3"](X.astype(np.float64), ref["X"][ok].astype(np.float64))
            if sim is not None:
                sim3[t] = float(sim[0])
        # Probe-verified direction: with G_{0<-t}, o_t = G[:3,3] and o_0 must be 0.
        assert np.linalg.norm(G[0, :3, 3]) < 1e-3, "G convention broken: o_0 != 0"
        self._K, self._G, self._sim3 = K, G, sim3
        return K, G, sim3

    # ---------------------------------------------------------------- depth
    def depth_map(self, t: int, stride: int = 1) -> np.ndarray:
        """z-depth (P3) of frame t at process res, model scale. Queried on a
        `stride` grid, bilinearly upsampled if stride > 1 (in provenance)."""
        import cv2
        Hm, Wm = self.process_hw
        us = np.arange(0, Wm, stride, dtype=np.float32)
        vs = np.arange(0, Hm, stride, dtype=np.float32)
        uv = np.stack(np.meshgrid(us, vs), -1).reshape(-1, 2)
        loc = self.query_global(uv, t, t, t)
        z = loc["X"][:, 2].reshape(len(vs), len(us))
        if stride > 1:
            z = cv2.resize(z, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
        return z.astype(np.float16)

    # ---------------------------------------------------------------- misc
    def provenance(self) -> dict:
        torch = self.torch
        try:
            d4rt_sha = subprocess.run(
                ["git", "-C", self.cfg["paths"]["d4rt_dir"], "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True).stdout.strip() or "unknown"
        except OSError:
            d4rt_sha = "unknown"
        env_file = Path("/content/env_summary.txt")
        return {
            "geometry_source": "opend4rt-adapter (single source, A-G8)",
            "model_commit": d4rt_sha,
            "checkpoint": self.checkpoint_name,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "dtype": self.cfg["d4rt"]["dtype"],
            "process_hw": list(self.process_hw),
            "native_hw": list(self.native_hw),
            "resize_factors": self.resize_factors,
            "clip_frames_window": self.clip_frames,
            "depth_stride": int(self.cfg["video"]["depth_stride"]),
            "conventions": {"G": "G_{0<-t} (P2)", "depth": "z-depth (P3)",
                            "K": "lsq_pinhole_fit per frame (P5)"},
            "x0_window_stitch": {
                "method": "overlap sim3 -> reference window (signed 2026-08-02)",
                "scales": {f"win@{w[1]}": round(g[0], 4)
                           for w, g in self._stitch_cache.items() if g[0] is not None}},
            "env": env_file.read_text(encoding="utf-8").strip() if env_file.exists() else "n/a",
        }
