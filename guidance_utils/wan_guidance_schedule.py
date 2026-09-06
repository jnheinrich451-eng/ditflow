"""Sampling-index schedules, independent of scheduler timestep values."""

import numpy as np


def window_indices(num_steps, window):
    """[50, 40] selects indices 0..9 out of 50; [40, 20] selects 10..29."""
    if len(window) != 2:
        raise ValueError('A timestep range requires two integers: MAX MIN')
    high, low = window
    if not 0 <= low <= high or low > num_steps:
        raise ValueError(f'Invalid timestep range {list(window)} for {num_steps} sampling steps')
    return list(range(max(num_steps - high, 0), num_steps - low))


def learning_rates(indices, endpoints, decay_steps=None):
    """Map absolute sampling indices to LRs, decaying over active updates.

    A fixed decay_steps lets extended windows share the baseline's LR prefix;
    subsequent active steps hold the final LR.
    """
    if len(endpoints) != 2 or not all(np.isfinite(v) and v >= 0 for v in endpoints):
        raise ValueError('lr requires two finite, nonnegative values')
    if decay_steps is not None and decay_steps < 1:
        raise ValueError('lr_decay_steps must be positive')
    if not indices:
        return {}
    count = len(indices) if decay_steps is None else decay_steps
    ramp = np.linspace(*endpoints, count)
    return {step: float(ramp[min(i, count - 1)]) for i, step in enumerate(indices)}
