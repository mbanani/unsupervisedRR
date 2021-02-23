from typing import List, Optional

import torch
from torch import nn as nn


# weigh initialization from torchvision/models/vgg.py
def initialize_weights(network):
    for m in network.modules():
        # add deconv to instances?
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.zero_()
        elif "activation" in str(type(m)) or "container" in str(type(m)):
            pass
        elif "pooling" in str(type(m)) or "upsampling" in str(type(m)):
            pass
        else:
            print("Warning: {} not handled or initialized".format(type(m)))


# Code to convert a specific layer in a network with another; useful for updating ReLU
# to LeakyReLU or removing batchnorm.
# Based on answer on PyTorch Discussion by @mostafa_elhoushi
def convert_layers(model, old_layer, new_layer, layer_args={}):
    count = 0
    for name, module in reversed(model._modules.items()):
        if type(module) == old_layer:
            model._modules[name] = new_layer(**layer_args)
            count += 1
        elif len(list(module.children())) > 0:
            # recurse
            mod, r_count = convert_layers(module, old_layer, new_layer, layer_args)
            model._modules[name] = mod
            count += r_count

    return model, count


def change_padding(model, layer, new_padding):
    for m in model.modules():
        if isinstance(m, layer):
            m.padding_mode = new_padding


# Code for freezing batchnorm
def freeze_bn(network):
    for m in network.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()


def unfreeze_bn(network):
    for m in network.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.train()


def get_grid(B: int, H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    grid_xyz = grid_xyz[None, :, :, :].repeat(B, 1, 1, 1)

    return grid_xyz


@torch.jit.script
def grid_to_pointcloud(
    K_inv,
    depth,
    features: Optional[torch.Tensor],
    grid: Optional[torch.Tensor],
    downsampled: bool = False,
    match_feats: bool = False,
):
    if match_feats and features is not None:
        B, _, H, W = features.shape
        depth = nn.functional.interpolate(
            depth, (H, W), mode="bilinear", align_corners=True
        )
    else:
        B, _, H, W = depth.shape

    if grid is None:
        grid = get_grid(B, H, W)

    # Apply inverse projection
    points = depth * grid

    if downsampled:
        points = nn.functional.avg_pool2d(points, 2, 2)
        if features is not None:
            features = nn.functional.avg_pool2d(features, 2, 2)
        H = H // 2
        W = W // 2

    # Invert intriniscs
    points = points.view(B, 3, H * W)
    points = K_inv.bmm(points)
    points = points.permute(0, 2, 1)

    if features is not None:
        # convert
        features = features.view(B, features.shape[1], H * W)
        features = features.permute(0, 2, 1)

    return points, features


def nn_gather(points, indices):
    # expand indices to same dimensions as points
    indices = indices[:, :, None]
    indices = indices.expand(indices.shape[0], indices.shape[1], points.shape[2])
    return points.gather(1, indices)


@torch.jit.script
def points_to_ndc(pts, K, img_dim: List[float], renderer: bool = True):
    pts = pts.bmm(K.transpose(1, 2))

    x = pts[:, :, 0:1]
    y = pts[:, :, 1:2]
    z = pts[:, :, 2:3]

    # remove very close z)
    z_min = 1e-5
    z = z.clamp(z_min)
    # z = torch.where(z > 0, z.clamp(z_min), z - 1)

    x = 2.0 * (x / z / img_dim[1]) - 1.0
    y = 2.0 * (y / z / img_dim[0]) - 1.0

    # apply negative for the pytorch3d renderer
    if renderer:
        ndc = torch.cat((-x, -y, z), dim=2)
    else:
        ndc = torch.cat((x, y, z), dim=2)
    return ndc
