import numpy as np
import torch
from pytorch3d.transforms import so3_relative_angle, so3_rotation_angle
from torch.nn import functional as F

from .transformations import transform_points_Rt


def evaluate_pose_Rt(pr, gt, scaled=False):
    assert pr.shape[1] == 3 and pr.shape[2] == 4, "pose should be Rt (3x4)"
    assert pr.shape == gt.shape

    pr_t = pr[:, :, 3:4]
    pr_R = pr[:, :, 0:3]
    gt_t = gt[:, :, 3:4]
    gt_R = gt[:, :, 0:3]

    # compute rotation error
    R_error = so3_relative_angle(pr_R, gt_R)
    R_mag = so3_rotation_angle(gt_R)
    # convert to degrees
    R_error = R_error * 180.0 / np.pi
    R_mag = R_mag * 180.0 / np.pi

    # calculate absolute xyz error
    t_abs_error = (pr_t - gt_t).norm(p=2, dim=1)
    t_mag = gt_t.norm(p=2, dim=1)
    t_ang_error = F.cosine_similarity(pr_t, gt_t, 1, 1e-8).acos()
    # convert to cms
    t_mag = t_mag * 100.0
    t_abs_error = t_abs_error * 100.0
    t_ang_error = t_ang_error * 180.0 / np.pi

    output = {
        "vp-mag_R": R_mag,
        "vp_mag_t": t_mag,
        "vp-error_R": R_error,
        "vp-error_t": t_abs_error,
        "vp-error_t-radian": t_ang_error,
    }

    # calculate relative scaled translation error
    if scaled:
        scaling = (gt_t * pr_t).sum(dim=1) / (gt_t * gt_t).sum(dim=1).clamp(min=1e-9)
        gt_t_scaled = gt_t * scaling[:, None]
        t_rel_error = (pr_t - gt_t_scaled).norm(p=2, dim=1)

        output["vp-error_t-scaled"] = t_rel_error

    return output


def evaluate_correspondances(pixel_loc_0, pixel_loc_1, depth_0, K, Rt):
    """
    Inputs:
        pixel_loc_0     LongTensor      batch_size x 2 x num_matches
        pixel_loc_1     LongTensor      batch_size x 2 x num_matches
        depth_0         FloatTensor     batch_size x 1 x num_matches
        Rt              FloatTensor     batch_size x 3 x 4

    Output:
        corr_error      FloatTensor     batch_size x num_matches
    """
    # (u, v) + depth_0 + K -> xyz_0 (B, 3, M)
    pixel_loc_0 = pixel_loc_0 * depth_0
    uvd_0 = torch.cat((pixel_loc_0, depth_0), dim=1)
    xyz_0 = K.inverse().bmm(uvd_0)

    # xyz_0 and Rt -> xyz_1
    R, t = Rt[:, :, :3], Rt[:, :, 3]
    xyz_1 = R.bmm(xyz_0) + t[:, :, None]

    # xyz_1 and K -> (u, v)_1 pred
    uvd_1 = K.bmm(xyz_1)
    uv_1 = uvd_1[:, :2, :]
    d_1 = uvd_1[:, 2:3, :]
    pred_pixel_loc_1 = uv_1 / d_1.clamp(min=1e-9)

    # get error in pixel values for (u, v) 1
    corr_error = (pred_pixel_loc_1 - pixel_loc_1).norm(dim=1, p=2)

    # handle invalid depth
    invalid = depth_0[:, 0] <= 0.0
    corr_error[invalid] = -1.0

    return corr_error


def evaluate_3d_correspondances(xyz_0, xyz_1, K, Rt, img_dim):
    """
    Inputs:
        xyz_0     FloatTensor       batch_size x 3 x num_matches
        xyz_1     FloatTensor       batch_size x 3 x num_matches
        K         FloatTensor       batch_size x 3 x 3
        Rt        FloatTensor       batch_size x 3 x 4
        img_dim   tuple(int, int)   (image height, image width)

    Output:
        corr_error      FloatTensor     batch_size x num_matches
    """
    xyz_0in1 = transform_points_Rt(xyz_0, Rt, inverse=False)
    xyz_1in0 = transform_points_Rt(xyz_1, Rt, inverse=True)

    def xyz_to_camera(xyz):
        uvd = xyz.bmm(K.transpose(1, 2))
        return uvd[:, :, :2] / uvd[:, :, 2:3]

    xyz_all_0 = torch.cat((xyz_0, xyz_0in1), dim=1)
    xyz_all_1 = torch.cat((xyz_1in0, xyz_1), dim=1)

    uv_all_0 = xyz_to_camera(xyz_all_0)
    uv_all_1 = xyz_to_camera(xyz_all_1)

    uv_0 = xyz_to_camera(xyz_0)
    uv_1 = xyz_to_camera(xyz_1)

    H, W = img_dim

    valid_dep = (xyz_all_0[:, :, 2] > 0) & (xyz_all_1[:, :, 2] > 0)
    valid_x_l = (uv_all_0[:, :, 0] > 0) & (uv_all_1[:, :, 0] > 0)
    valid_x_h = (uv_all_0[:, :, 0] < W) & (uv_all_1[:, :, 0] < W)
    valid_y_l = (uv_all_0[:, :, 1] > 0) & (uv_all_1[:, :, 1] > 0)
    valid_y_h = (uv_all_0[:, :, 1] < H) & (uv_all_1[:, :, 1] < H)

    valid = torch.stack((valid_dep, valid_x_l, valid_x_h, valid_y_l, valid_y_h), dim=2)

    valid = valid.all(dim=2).float()
    valid_denom = valid.sum(dim=1).clamp(min=1)
    valid_points = valid.mean(dim=1)

    error_2d = (uv_all_0 - uv_all_1).norm(dim=2, p=2)
    error_2d_mean = (error_2d * valid).sum(dim=1) / valid_denom

    error_3d = (xyz_all_0 - xyz_all_1).norm(dim=2, p=2)

    output = {
        "corr3d-meanError": error_3d.mean(dim=1),
        "corr2d-meanError": error_2d_mean,
        "corr2d-validPoints": valid_points,
    }

    for pix_thresh in [1, 2, 5]:
        inlier_thresh = error_2d <= pix_thresh
        inlier_percent = (inlier_thresh.float() * valid).sum(dim=1) / valid_denom
        output[f"corr2d-within{pix_thresh}px"] = inlier_percent

    return output, (uv_0, uv_1)
