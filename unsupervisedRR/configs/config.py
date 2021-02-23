"""
YACS based configuration file (https://github.com/rbgirshick/yacs)
"""
import os

from yacs.config import CfgNode as CN

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_PATH = os.path.dirname(PROJECT_PATH)

_C = CN()

_C.SYSTEM = CN()
_C.PATHS = CN()
_C.TRAIN = CN()
_C.MODEL = CN()
_C.DATASET = CN()
_C.EXPERIMENT = CN()

# System Parameters
_C.SYSTEM.RANDOM_SEED = 8
_C.SYSTEM.NUM_WORKERS = 6
_C.SYSTEM.TQDM = True

# Paths
_C.PATHS.project_root = PROJECT_PATH
_C.PATHS.html_visual_dir = ""  # Not being used.
_C.PATHS.tensorboard_dir = os.path.join(REPO_PATH, "logs", "tensor_logs")
_C.PATHS.experiments_dir = os.path.join(REPO_PATH, "logs", "experiments")

# Training Parameters
_C.TRAIN.num_epochs = 100
_C.TRAIN.eval_step = 5000
_C.TRAIN.vis_step = 500
_C.TRAIN.lr = 1e-4
_C.TRAIN.momentum = 0.9
_C.TRAIN.weight_decay = 1e-6
_C.TRAIN.rgb_render_loss_weight = 1.0
_C.TRAIN.rgb_decode_loss_weight = 0.0
_C.TRAIN.depth_loss_weight = 1.0
_C.TRAIN.correspondance_loss_weight = 0.1
_C.TRAIN.optimizer = "Adam"
_C.TRAIN.scheduler = "constant"
_C.TRAIN.resume = ""  # Checkpoint path to resume training

# Dataset Parameters
_C.DATASET.name = ""
_C.DATASET.num_views = 2
_C.DATASET.view_spacing = 20
_C.DATASET.img_dim = 128
# -- loader params
_C.DATASET.num_workers = 4
_C.DATASET.overfit = False
_C.DATASET.batch_size = 8

# Model Parameters
_C.MODEL.name = "DEFINED BY MODEL"
_C.MODEL.feat_dim = 32
_C.MODEL.num_views = 2
_C.MODEL.use_gt_vp = False
# -- define renderer (PyTorch3D defaults; skipping bin settings)
_C.MODEL.renderer = CN()
_C.MODEL.renderer.render_size = 128
_C.MODEL.renderer.pointcloud_source = "other"
_C.MODEL.renderer.radius = 2.0
_C.MODEL.renderer.points_per_pixel = 16
_C.MODEL.renderer.compositor = "norm_weighted_sum"
_C.MODEL.renderer.weight_calculation = "exponential"
# -- define alignment
_C.MODEL.alignment = CN()
_C.MODEL.alignment.algorithm = "weighted_procrustes"
_C.MODEL.alignment.base_weight = "nn_ratio"
_C.MODEL.alignment.num_correspodances = 200  # This is actually half the correspondances
_C.MODEL.alignment.point_ratio = 0.2
_C.MODEL.alignment.num_seeds = 10

# Experiment Parameters
_C.EXPERIMENT.name = ""
_C.EXPERIMENT.rationale = ""
_C.EXPERIMENT.just_evaluate = False
_C.EXPERIMENT.checkpoint = ""


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
