"""Experimental reference-region weighting; not the faithful DiTFlow baseline."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_reference_regions(directory, num_frames, latent_frames, h, w, device):
    """Reference-coordinate foreground for every ordered latent pair.

    Anchors 0,4,...,20 approximate VAE support, matching the offline audit.
    This does not locate or retarget the generated subject.
    """
    directory = Path(directory)
    files = sorted(directory.glob('*.png'))
    expected = [f'{i:05d}.png' for i in range(num_frames)]
    if [p.name for p in files] != expected:
        raise ValueError(f'{directory}: expected exactly aligned masks 00000.png through {num_frames-1:05d}.png')
    if latent_frames < 2 or (num_frames-1) % (latent_frames-1):
        raise ValueError('Cannot map reference frames to temporal latent anchors')
    stride = (num_frames-1)//(latent_frames-1)
    anchors = list(range(0,num_frames,stride))
    masks = []
    for index in anchors:
        array = np.asarray(Image.open(files[index]))
        if array.ndim != 2:
            raise ValueError('Use single-channel or palette-index reference masks')
        coverage = np.asarray(Image.fromarray((array>0).astype(np.float32)).resize((w,h),Image.Resampling.BOX))
        masks.append(coverage>=.1)
    # Source-major ordering: a source-frame region applies to every target frame.
    region = torch.as_tensor(np.stack(masks),device=device,dtype=torch.bool).reshape(latent_frames,h*w)
    pairs = region[:,None,:].expand(latent_frames,latent_frames,h*w).reshape(latent_frames**2,h*w)
    provenance = dict(variant='experimental_reference_region_balance',
        region_weights=dict(subject=.5,background=.5),anchor_frames=anchors,
        coverage_threshold=.1,coordinate_system='reference source-frame screen coordinates',
        masks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return pairs, provenance


def region_balanced_mse(prediction, reference, valid, foreground):
    """Give each nonempty region half the loss; preserve the original validity mask."""
    if prediction.shape != reference.shape or prediction.shape[:-1] != valid.shape or foreground.shape != valid.shape:
        raise ValueError('Prediction, target, validity and region shapes do not agree')
    subject, background = valid.bool() & foreground.bool(), valid.bool() & ~foreground.bool()
    if not subject.any() or not background.any():
        raise ValueError('Region balancing requires retained AMF entries in both subject and background')
    fg_loss = F.mse_loss(prediction[subject],reference[subject])
    bg_loss = F.mse_loss(prediction[background],reference[background])
    total = .5*fg_loss+.5*bg_loss
    return total, dict(subject_mse=fg_loss.detach(),background_mse=bg_loss.detach(),
                      subject_entries=subject.sum(),background_entries=background.sum())
