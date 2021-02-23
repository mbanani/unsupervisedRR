from torch import nn as nn


def get_rgb_loss(pred, gt, cover=None):
    pixel_loss = nn.functional.l1_loss(pred, gt, reduction="none")

    if cover is not None:
        assert len(cover.shape) == 4
        cover = cover.float()
        cover_weight = cover.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
        pixel_loss = pixel_loss * cover / cover_weight

    return pixel_loss.mean(dim=(1, 2, 3)), pixel_loss
