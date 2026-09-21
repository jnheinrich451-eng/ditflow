"""Stage C presence policy (SKILL §5, honest_absent): absence is labeled,
never implied by an empty mask alone."""
from __future__ import annotations

import numpy as np


def presence(masks, core, Q, n_fr, n_pos_min):
    """object_present[t] = enough valid subject tracks AND nonempty mask.
    Outside the [first, last] present span, masks are zeroed (no pre-entry
    hallucination)."""
    pres = np.zeros(n_fr, bool)
    for t in range(n_fr):
        pres[t] = (int(Q[core, t].sum()) >= int(n_pos_min)
                   and t in masks and bool(masks[t].any()))
    if pres.any():
        lo, hi = int(np.argmax(pres)), n_fr - 1 - int(np.argmax(pres[::-1]))
        for t in range(n_fr):
            if (t < lo or t > hi) and t in masks:
                masks[t] = np.zeros_like(masks[t])
                pres[t] = False
    return pres, masks
