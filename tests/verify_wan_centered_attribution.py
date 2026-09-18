"""Offline checks of attribution accounting; no model or motion-quality claim."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.wan_centered_attribution import (decompose, regions, native_frame_mass,
    mass_queries, reference_statistics, response_to_target)


def main():
    # Unequal region populations deliberately separate regional mean from global contribution.
    before = np.zeros((2, 4, 2))
    after = np.zeros_like(before)
    reference = np.ones_like(before)
    after[:, 0] = 1  # Improve torso completely.
    after[:, 1] = -1  # Worsen the fence; aggregate loss must expose this.
    valid = np.ones((2, 4), bool)
    valid[1, 3] = False  # Excluded points never contribute.
    roi = dict(torso=np.array([1, 0, 0, 0], bool), fence=np.array([0, 1, 0, 0], bool),
               other=np.array([0, 0, 1, 1], bool))
    result = decompose(before, after, reference, valid, roi)
    for stage in ('before', 'after'):
        np.testing.assert_allclose(sum(v['contribution_'+stage] for v in result['regions'].values()),
                                   result['loss_'+stage], rtol=0, atol=1e-14)
    assert result['regions']['torso']['reduction_contribution'] > 0
    assert result['regions']['fence']['reduction_contribution'] < 0
    assert result['loss_after'] > result['loss_before']
    assert result['regions']['other']['valid_count'] == 3
    partition = regions()
    assert np.stack(list(partition.values())).sum(axis=0).min() == 1
    assert np.stack(list(partition.values())).sum(axis=0).max() == 1
    bad = dict(roi, other=np.ones(4, bool))
    try:
        decompose(before, after, reference, valid, bad)
    except ValueError as exc:
        assert 'partition' in str(exc)
    else:
        raise AssertionError('Overlapping regions were accepted')
    # Different supports can manufacture apparent differences; common support must match.
    reverse_valid = valid.copy()
    reverse_valid[:, 0] = False
    stats = reference_statistics({'forward': (reference, valid), 'reverse': (-reference, reverse_valid)}, roi)
    assert stats['own_valid']['forward']['torso'][0]['valid_count'] == 1
    assert stats['common_valid']['forward']['torso'][0]['valid_count'] == 0
    assert stats['common_valid']['reverse']['torso'][0]['mean_xy'] is None
    response = response_to_target(before, after, reference, valid, roi)
    assert response['torso'][0]['patchwise_alignment_cosine'] > .999999
    assert response['fence'][0]['patchwise_alignment_cosine'] < -.999999
    for q in mass_queries():
        assert partition[q['region']][q['token_index'] % 1560]

    # Independent oracle: SDPA with per-frame one-hot V returns destination-frame mass.
    # Deliberately different heads ensure softmax(mean logits) cannot pass this check.
    import torch
    import torch.nn.functional as F
    q = torch.tensor([[[2., 0.], [-2., 0.]], [[1., 1.], [0., 2.]],
                      [[0., -1.], [1., 0.]], [[1., 0.], [1., 1.]]])
    k = torch.tensor([[[4., 0.], [4., 0.]], [[3., 0.], [3., 0.]],
                      [[-3., 0.], [-3., 0.]], [[-4., 0.], [-4., 0.]]])
    ids = [0, 2, 3]
    value = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]).expand(2, -1, -1)
    expected = F.scaled_dot_product_attention(q[ids].permute(1, 0, 2), k.permute(1, 0, 2), value)
    mass = native_frame_mass(q, k, ids, frames=2, spatial=2, chunk_size=1)
    np.testing.assert_allclose(mass, expected.permute(1, 0, 2).numpy(), atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(mass.sum(axis=-1), 1., atol=1e-6)
    assert mass[0, 0, 1] < .001 and mass[0, 1, 1] > .999
    print('PASS: global/regional loss accounting, common-mask coverage, target responses, '
          'fixed query placement, and per-head native mass against independent SDPA output.')


if __name__ == '__main__':
    main()
