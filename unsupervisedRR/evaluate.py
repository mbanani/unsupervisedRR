import argparse
import os
import pickle
import random

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from .configs import get_cfg_defaults
from .datasets import build_loader
from .models import build_model
from .models.model_util import get_grid
from .utils.metrics import evaluate_correspondances, evaluate_pose_Rt

# deterministic evaluation due to sampling in some methods
seed = 77
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Set path for where to save the output dictionaries
RESULTS_DIR = None


def evaluate_split(model, data_loader, args, dict_name=None, use_tqdm=True):
    all_metrics = {}
    all_outputs = {}

    for batch in tqdm(data_loader, disable=not use_tqdm, dynamic_ncols=True):
        batch_output, batch_metrics = forward_batch(model, batch)
        for metric in batch_metrics:
            b_metric = batch_metrics[metric].detach().cpu()
            if metric in all_metrics:
                all_metrics[metric] = torch.cat((all_metrics[metric], b_metric), dim=0)
            else:
                all_metrics[metric] = b_metric

        instances = batch_metrics["instance_id"]
        for ins in instances:
            all_outputs[ins] = {"Rt": batch_output["vp_1"].detach().cpu()}
            if "corres_01" in batch_output:
                _corres = batch_output["corres_01"]
                _corres = [_c.detach().cpu() for _c in _corres]
                all_outputs[ins]["corres"] = _corres

    # Save outputs
    if dict_name is not None:
        dict_path = os.path.join(RESULTS_DIR, dict_name)
        with open(dict_path, "wb") as handle:
            output_dict = {
                "metrics": all_metrics,
                "outputs": all_outputs,
                "args": args,
            }

            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save metrics
    for metric in all_metrics:
        if metric == "instance_id":
            continue
        vals = all_metrics[metric]
        summary = f"{metric:30s}: {vals.mean():7.3f} +/- {vals.std():7.3f}   || "
        summary += f"median {vals.median():7.3f}"
        print(summary)

    # calculate percentage under errors
    r_acc = []
    t_acc = []
    c_acc = []
    r_err = all_metrics["vp-error_R"]
    t_err = all_metrics["vp-error_t"]
    c_err = all_metrics["chamfer"] * 1000

    for error in [5, 10, 45]:
        r_acc.append((r_err <= error).float().mean().item())

    for error in [5, 10, 25]:
        t_acc.append((t_err <= error).float().mean().item())

    for error in [1, 5, 10]:
        c_acc.append((c_err <= error).float().mean().item())

    r_acc_str = "  |  ".join([f"{x * 100:4.1f}" for x in r_acc])
    t_acc_str = "  |  ".join([f"{x * 100:4.1f}" for x in t_acc])
    c_acc_str = "  |  ".join([f"{x * 100:4.1f}" for x in c_acc])
    print(f"Rotation Accuracies:        {r_acc_str}")
    print(f"Translation Accuracies:     {t_acc_str}")
    print(f"Chamfer Accuracies:         {c_acc_str}")

    print("For latex: ")
    latex = f"{r_acc[0] * 100:4.1f} & "
    latex += f"{r_acc[1] * 100:4.1f} & "
    latex += f"{r_acc[2] * 100:4.1f} & "
    latex += f"{r_err.mean():4.1f} & "
    latex += f"{r_err.median():4.1f} & "

    latex += f"{t_acc[0] * 100:4.1f} & "
    latex += f"{t_acc[1] * 100:4.1f} & "
    latex += f"{t_acc[2] * 100:4.1f} & "
    latex += f"{t_err.mean():4.1f} & "
    latex += f"{t_err.median():4.1f} & "

    latex += f"{c_acc[0] * 100:4.1f} & "
    latex += f"{c_acc[1] * 100:4.1f} & "
    latex += f"{c_acc[2] * 100:4.1f} & "
    latex += f"{c_err.mean():4.1f} & "
    latex += f"{c_err.median():4.1f} & "

    print(latex)


def forward_batch(model, batch):
    num_views = 2

    gt_rgb = [batch[f"rgb_{i}"].cuda() for i in range(num_views)]
    gt_dep = [batch[f"depth_{i}"].cuda() for i in range(num_views)]
    gt_vps = [batch[f"Rt_{i}"].cuda() for i in range(num_views)]
    K = batch["K"].cuda()

    output = model(gt_rgb, K=K, deps=gt_dep)

    metrics = {"instance_id": batch["uid"]}

    # Model outputs
    vp_1 = output["vp_1"]
    pr_pc = output["joint_pointcloud"]
    gt_pc = model.generate_pointclouds(K, gt_dep, gt_vps)

    # Evaluate pose
    p_metrics = evaluate_pose_Rt(vp_1, gt_vps[1])
    for _k in p_metrics:
        metrics[f"{_k}"] = p_metrics[_k].detach().cpu()

    # get chamfer metrics
    cham = chamfer_distance(pr_pc.cuda(), gt_pc.cuda(), batch_reduction=None)[0].cpu()
    metrics["chamfer"] = cham

    # gather inputs
    if "corres_01" in output:
        id_c0, id_c1, c_ratio, _ = output["corres_01"]
        # Evaluate correspondaces -- should REALLY be factored out more
        depth_0 = gt_dep[0]
        B, _, H, W = depth_0.shape

        depth_0 = depth_0.view(B, 1, -1)
        id_01_0 = id_c0.unsqueeze(1)
        id_01_1 = id_c1.unsqueeze(1)

        grid = get_grid(B, H, W)
        grid = grid[:, :2].view(B, 2, -1).to(depth_0.device)
        dep01_0 = depth_0.gather(2, id_01_0)

        pix01_0 = grid.gather(2, id_01_0.repeat(1, 2, 1))
        pix01_1 = grid.gather(2, id_01_1.repeat(1, 2, 1))

        Rt_i = gt_vps[1]
        c_err_i = evaluate_correspondances(pix01_0, pix01_1, dep01_0, K, Rt_i)

        # errors cannot be larger than diagnonal (impossible .. )
        diag = (H ** 2 + W ** 2) ** 0.5
        c_err_i = c_err_i.clamp(max=diag)

        valid = (c_err_i >= 0).float()
        valid_denom = valid.sum(dim=1).clamp(min=1)
        error = (c_err_i * valid).sum(dim=1) / valid_denom
        metrics["corr-validDepth"] = valid.mean(dim=1) * 100.0
        metrics["corr-meanError"] = error

        for px_thresh in [2, 4, 10]:
            in_px = (c_err_i < px_thresh).float()
            in_px = (in_px * valid).sum(dim=1) / valid_denom
            metrics[f"corr-within{px_thresh}px"] = in_px * 100.0

    return output, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="ScanNet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--boost_alignment", default=False, action="store_true")
    parser.add_argument("--save_dict", type=str, default=None)
    parser.add_argument("--progress_bar", default=False, action="store_true")
    parser.add_argument("--no_ratio", default=False, action="store_true")
    parser.add_argument("--point_ratio", default=None, type=float)
    parser.add_argument("--num_seeds", default=None, type=int)
    args = parser.parse_args()

    # Dataset configs to be decided
    default_cfg = get_cfg_defaults()
    default_cfg.defrost()

    # Dataset Parameters
    dataset_cfg = default_cfg.DATASET
    dataset_cfg.name = args.dataset
    dataset_cfg.batch_size = 4
    data_loader = build_loader(dataset_cfg, split=args.split)

    # Define model
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model_weights = checkpoint["model"]
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Step:  {checkpoint['step']}")

        # Load checkpoint
        model_cfg = checkpoint["cfg"].MODEL
        model_cfg.defrost()

    else:
        model_cfg = default_cfg.MODEL
        model_weights = None
        model_cfg.name = args.model

    # Set alignmnet performance
    if args.boost_alignment:
        assert not args.no_ratio
        assert args.num_seeds is None
        assert args.point_ratio is None

        model_cfg.alignment.defrost()
        model_cfg.alignment.num_seeds = 100
        model_cfg.alignment.point_ratio = 0.05
        model_cfg.alignment.base_weight = "nn_ratio"

    if args.no_ratio:
        model_cfg.alignment.base_weight = "uniform"

    if args.num_seeds is not None:
        model_cfg.alignment.num_seeds = args.num_seeds

    if args.point_ratio is not None:
        model_cfg.alignment.point_ratio = args.point_ratio

    model = build_model(model_cfg).cuda()

    if model_weights is not None:
        model.load_state_dict(model_weights)

    evaluate_split(model, data_loader, args, args.save_dict, use_tqdm=args.progress_bar)
