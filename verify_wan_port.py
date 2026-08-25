"""Self-check for the Wan2.1 port. Run this before trusting a sweep.

Weight-free: builds a tiny randomly-initialised Wan transformer, so it needs no
checkpoint, no GPU and a few seconds. Architecture and autograd behaviour do not
depend on weights, which is the point -- this is cheap enough to run at the
start of any session that touches the port.

    python verify_wan_port.py

The AMF check compares against **this repo's own** `guidance_utils/
motion_flow_utils.py` -- the published DiTFlow implementation -- not against
another copy of the port, so it is a real independent check rather than a
tautology. The port computes the same quantity via three algebraic rewrites
(head fusion, frame-pair chunking, displacement without relative-coordinate
grids); see the module docstring in `guidance_utils/wan_motion_flow_utils.py`.

Exits non-zero on any failure.
"""

import sys
import traceback

import torch
import torch.nn.functional as F

from guidance_utils.motion_flow_utils import compute_motion_flow as reference_amf
from guidance_utils.wan_motion_flow_utils import compute_motion_flow
from guidance_utils.wan_modules import WanInjectionProcessor, WanModuleWithGuidance
from guidance_utils.wan_transformer import ControlledWanTransformer

TINY = dict(
    patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
    in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32,
    num_layers=4, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
    eps=1e-6, rope_max_seq_len=64,
)
TEXT_PREFIX = 226  # what the CogVideoX reference slices off; Wan has no prefix

FAILURES = []


def check(name, detail, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def check_amf():
    """Port == published DiTFlow AMF, at f64."""
    print("\n=== AMF equivalence vs guidance_utils/motion_flow_utils.py ===")
    torch.manual_seed(0)
    h, w, f, heads, dim = 5, 7, 3, 4, 8
    seq = f * h * w

    # Reference wants (B, heads, TEXT+S, D); the port wants (B, S, heads, D).
    q_ref = torch.randn(2, heads, TEXT_PREFIX + seq, dim, dtype=torch.float64)
    k_ref = torch.randn(2, heads, TEXT_PREFIX + seq, dim, dtype=torch.float64)
    q_wan = q_ref[:, :, TEXT_PREFIX:, :].permute(0, 2, 1, 3).contiguous()
    k_wan = k_ref[:, :, TEXT_PREFIX:, :].permute(0, 2, 1, 3).contiguous()

    for argmax in (False, True):
        for temp in (2.0, 5.0):
            a = reference_amf(q_ref, k_ref, h=h, w=w, temp=temp, nframes=f, argmax=argmax)
            b = compute_motion_flow(q_wan, k_wan, h=h, w=w, nframes=f, temp=temp,
                                    argmax=argmax, softmax_fp32=False)
            err = (a.double() - b.double()).abs().max().item()
            check(f"amf argmax={argmax!s:5s} temp={temp:g}",
                  f"max|diff| = {err:.3e}", a.shape == b.shape and err < 1e-11)

    # Frame-pair recomputation must change neither value nor gradients.
    grads = []
    for ckpt in (False, True):
        qg = q_wan.clone().float().requires_grad_(True)
        kg = k_wan.clone().float().requires_grad_(True)
        compute_motion_flow(qg, kg, h=h, w=w, nframes=f, temp=2.0,
                            checkpoint_pairs=ckpt).sum().backward()
        grads.append((qg.grad, kg.grad))
    dq = (grads[0][0] - grads[1][0]).abs().max().item()
    dk = (grads[0][1] - grads[1][1]).abs().max().item()
    check("amf checkpoint_pairs invariant", f"dq={dq:.3e} dk={dk:.3e}", dq == 0.0 and dk == 0.0)


def check_chain():
    """Transformer subclass, QK capture, gradients, checkpointing, injection."""
    print("\n=== transformer / guidance chain ===")
    torch.manual_seed(0)
    model = ControlledWanTransformer(**TINY).eval()
    for i in range(len(model.blocks)):
        model.blocks[i].attn1.set_processor(WanInjectionProcessor(f"block_{i}_attn1_processor"))

    frames, height, width = 3, 8, 10
    ppf, pph, ppw = 3, 4, 5
    n = ppf * pph * ppw
    lat = torch.randn(1, TINY["in_channels"], frames, height, width)
    txt = torch.randn(1, 12, TINY["text_dim"])
    ts = torch.tensor([0])

    base = model(hidden_states=lat, timestep=ts, encoder_hidden_states=txt, return_dict=False)[0]
    check("full forward", f"{tuple(base.shape)}", base.shape == lat.shape)

    model.init_rope = model.default_rope(lat)
    rope_qk = torch.stack([model.init_rope, model.init_rope], dim=0)
    explicit = model(hidden_states=lat, timestep=ts, encoder_hidden_states=txt,
                     rope=rope_qk, return_dict=False)[0]
    d = (base - explicit).abs().max().item()
    check("rotary rewrite identical to diffusers", f"max|diff| = {d:.3e}", d == 0.0)

    gb = 2
    model.blocks[gb] = WanModuleWithGuidance(model.blocks[gb], height, width, 2, "block_2", ppf)
    proc = model.blocks[gb].attn1.processor
    proc.copy_kv = True
    model.stop_after_block = gb
    early = model(hidden_states=lat, timestep=ts, encoder_hidden_states=txt, return_dict=False)[0]

    check("Q/K layout (B,S,heads,D), no text prefix",
          f"{tuple(proc.query.shape)}, S={proc.query.shape[1]}=={n}",
          proc.query.shape == (1, n, TINY["num_attention_heads"], TINY["attention_head_dim"]))
    check("early exit", f"{tuple(early.shape)} token-space", early.shape == (1, n, 16))
    check("feature hook (t,d,h,w)", f"{tuple(model.blocks[gb].saved_features.shape)}",
          model.blocks[gb].saved_features.shape == (ppf, 16, pph, ppw))

    target = compute_motion_flow(proc.query.detach(), proc.key.detach(),
                                 h=pph, w=ppw, nframes=ppf, temp=2.0, argmax=True)

    def guidance_loss(**kw):
        proc.clear()
        model(timestep=ts, encoder_hidden_states=txt, return_dict=False, **kw)
        amf = compute_motion_flow(proc.query, proc.key, h=pph, w=ppw, nframes=ppf, temp=2.0)
        return F.mse_loss(amf, target.to(amf.dtype))

    x = lat.clone().requires_grad_(True)
    guidance_loss(hidden_states=x).backward()
    check("gradient reaches latent", f"|grad| = {x.grad.abs().sum():.4e}", x.grad.abs().sum() > 0)

    opt_rope = rope_qk.clone().detach().requires_grad_(True)
    guidance_loss(hidden_states=lat, rope=opt_rope).backward()
    check("gradient reaches rope", f"|grad| = {opt_rope.grad.abs().sum():.4e}",
          opt_rope.grad.abs().sum() > 0)

    # Q/K are captured as a SIDE EFFECT inside a checkpointed region -- the
    # port's least obvious assumption, so assert gradients survive it.
    model.enable_gradient_checkpointing()
    x2 = lat.clone().requires_grad_(True)
    guidance_loss(hidden_states=x2).backward()
    gd = (x2.grad - x.grad).abs().max().item()
    check("gradient checkpointing invariant", f"max|diff| = {gd:.3e}", gd == 0.0)
    model.disable_gradient_checkpointing()

    model.stop_after_block = None
    inj = model.blocks[0].attn1.processor
    inj.copy_kv = True
    model(hidden_states=torch.randn_like(lat), timestep=ts, encoder_hidden_states=txt, return_dict=False)
    inj.copy_kv, inj.inject_kv = False, True
    injected = model(hidden_states=lat, timestep=ts, encoder_hidden_states=txt, return_dict=False)[0]
    inj.inject_kv = False
    delta = (injected - base).abs().max().item()
    check("KV injection effective", f"changes output by {delta:.3e}", delta > 1e-5)


def main():
    print(f"torch {torch.__version__}")
    for fn in (check_amf, check_chain):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            FAILURES.append(fn.__name__)

    if FAILURES:
        print(f"\nFAILED: {', '.join(FAILURES)}")
        return 1
    print("\nAll Wan port checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
