import torch
from pytorch3d import transforms as pt3d_T


@torch.jit.script
def invert_quaternion(quat):
    return quat * torch.tensor([1, -1, -1, -1]).to(quat)


@torch.jit.script
def normalize_quaternion(quat):
    # deal with all 0s
    norm = quat.norm(p=2, dim=-1, keepdim=True)
    w = quat[..., 0:1]
    w = torch.where(norm < 1e-9, w + 1, w)
    quat = torch.cat((w, quat[..., 1:]), dim=-1)

    # normalize
    norm = quat.norm(p=2, dim=-1, keepdim=True)
    quat = quat / quat.norm(p=2, dim=-1, keepdim=True)
    return quat


@torch.jit.script
def quaternion_distance(q0, q1):
    w_rel = (q0 * q1).sum(dim=1)
    w_rel = w_rel.clamp(min=-1, max=1)
    q_rel_error = 2 * w_rel.abs().acos()
    return q_rel_error


@torch.jit.script
def normalize_qt(params):
    t = params[:, 4:7]
    q = params[:, 0:4]
    q = normalize_quaternion(q)
    return torch.cat((q, t), dim=-1)


@torch.jit.script
def apply_quaternion(quaternion, point):
    # Copied over as is from pytorch3d; replaced wiht my invert
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = pt3d_T.quaternion_raw_multiply(
        pt3d_T.quaternion_raw_multiply(quaternion, point_as_quaternion),
        invert_quaternion(quaternion),
    )
    return out[..., 1:]


def random_qt(batch_size: int, q_mag: float, t_mag: float):
    assert q_mag >= 0.0 and q_mag < 1.57, "Rotation angle has to be between 0 and pi/2"

    # Random quaternion of magnitude theta (cos(theta/2
    h_mag = torch.ones(batch_size, 1) * q_mag / 2.0
    q_w = h_mag.cos()
    q_xyz = torch.randn(batch_size, 3)
    q_xyz = (q_xyz / q_xyz.norm(p=2, dim=1, keepdim=True)) * h_mag.sin()

    # get translation
    t = torch.randn(batch_size, 3)
    t = t / t.norm(p=2, dim=1, keepdim=True)

    param = torch.cat((q_w, q_xyz, t), dim=1)
    return param


@torch.jit.script
def transform_points_qt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, D = viewpoint.shape
    assert D == 7, "3 translation and 4 quat "

    q = viewpoint[:, None, 0:4]
    t = viewpoint[:, None, 4:7]

    # Normalize quaternion
    q = normalize_quaternion(q)

    if inverse:
        # translate then rotate
        points = points - t
        points = apply_quaternion(invert_quaternion(q), points)
    else:
        # rotate then invert
        points = apply_quaternion(q, points)
        points = points + t

    return points


@torch.jit.script
def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, H, W = viewpoint.shape
    assert H == 3 and W == 4, "Rt is B x 3 x 4 "
    t = viewpoint[:, :, 3]
    r = viewpoint[:, :, 0:3]

    # transpose r to handle the fact that P in num_points x 3
    # yT = (RX)T = XT @ RT
    r = r.transpose(1, 2).contiguous()

    # invert if needed
    if inverse:
        points = points - t[:, None, :]
        points = points.bmm(r.inverse())
    else:
        points = points.bmm(r)
        points = points + t[:, None, :]

    return points


if __name__ == "__main__":
    rand_pt = torch.randn(4, 1000, 3)

    rand_qt = random_qt(4, 0.5, 3)

    # qt -> rt
    q = rand_qt[:, :4]
    t = rand_qt[:, 4:, None]

    R = pt3d_T.quaternion_to_matrix(q)
    Rinv = pt3d_T.quaternion_to_matrix(pt3d_T.quaternion_invert(q))

    Rti = torch.cat((Rinv, t), dim=2)
    Rt = torch.cat((R, t), dim=2)

    rot_qt = transform_points_qt(rand_pt, rand_qt)
    rot_Rt = transform_points_Rt(rand_pt, Rt)
    rot_Rti = transform_points_Rt(rand_pt, Rti)

    qt_Rt = (rot_qt - rot_Rt).norm(dim=2, p=2).mean()
    qt_Rti = (rot_qt - rot_Rti).norm(dim=2, p=2).mean()
    Rt_Rti = (rot_Rti - rot_Rt).norm(dim=2, p=2).mean()

    print(f"|| points ||:    {rand_pt.norm(p=2,dim=2).mean()}")
    print(f"Diff Rt and qt:  {qt_Rt:.4e}")
    print(f"Diff Rti and qt: {qt_Rti:.4e}")
    print(f"Diff Rti and Rt: {Rt_Rti:.4e}")
