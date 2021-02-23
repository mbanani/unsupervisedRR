import os
import pickle

from torch.utils.data import DataLoader

from .video_dataset import VideoDataset

# Define some important paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RGBD_3D_ROOT = None  # -- You need to define those --
SCANNET_ROOT = None  # -- You need to define those --


def build_dataset(cfg, split, overfit=None):
    """
    Builds a dataset from the provided dataset configs.
    Configs can be seen is configs/config.py
    """

    if cfg.name == "ScanNet":
        root_path = SCANNET_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"data/scannet_{split}.pkl")
    elif cfg.name == "RGBD_3DMatch":
        root_path = RGBD_3D_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"data/3dmatch_{split}.pkl")
    else:
        raise ValueError("Dataset name {} not recognized.".format(cfg.name))

    with open(dict_path, "rb") as f:
        data_dict = pickle.load(f)
    dataset = VideoDataset(cfg, root_path, data_dict, split)

    # Reduce ScanNet validation size to allow for more frequent validation
    if cfg.name == "ScanNet" and split == "valid":
        dataset.instances = dataset.instances[::10]

    # Overfit only loads a single batch for easy debugging/sanity checks
    if overfit is not None:
        assert type(overfit) is int
        dataset.instances = dataset.instances[: cfg.batch_size] * overfit

    return dataset


def build_loader(cfg, split, overfit=None):
    """
    Builds the dataset loader (including getting the dataset).
    """
    dataset = build_dataset(cfg, split, overfit)
    shuffle = (split == "train") and (not overfit)
    batch_size = cfg.batch_size

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    return loader
