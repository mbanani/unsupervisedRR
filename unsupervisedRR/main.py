import argparse
import os
import shutil

import torch

from .configs import get_cfg_defaults
from .nnutils.pcreg_trainer import PCReg_Trainer

if __name__ == "__main__":
    # general parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", type=str, help="Experiment Mode: {train or test}.",
    )
    parser.add_argument(
        "config_path", type=str, help="Path for experiment config file.",
    )
    parser.add_argument(
        "--modifier", type=str, help="Path for modifier config file.", default=None,
    )
    parser.add_argument(
        "--debug", default=False, action="store_true",
    )
    parser.add_argument(
        "--overfit", default=False, action="store_true",
    )
    args = parser.parse_args()

    # ==== Setup Config File =====
    # load config
    cfg = get_cfg_defaults()
    config_path = os.path.join("unsupervisedRR/configs", args.config_path)
    cfg.merge_from_file(config_path)

    # deal with modifiers
    if args.modifier is not None:
        assert cfg.EXPERIMENT.name == "", "Modifier config defines experiment name."
        modifier_path = os.path.join("unsupervisedRR/configs", args.modifier)
        cfg.merge_from_file(modifier_path)

    # easy modifier for debugging
    if args.debug:
        cfg.EXPERIMENT.name = "DEBUG"
        cfg.SYSTEM.TQDM = True
        torch.autograd.set_detect_anomaly(True)

        # Remove debug tensor log
        log_path = os.path.join(cfg.PATHS.tensorboard_dir, "DEBUG")
        vis_path = os.path.join(cfg.PATHS.html_visual_dir, "DEBUG")
        try:
            shutil.rmtree(log_path)
            shutil.rmtree(vis_path)
        except:
            pass

    if args.overfit:
        cfg.DATASET.overfit = True
        cfg.SYSTEM.TQDM = True

    cfg.freeze()

    assert cfg.EXPERIMENT.name != "", "Experiment name is not defined."

    print("=====================================")
    print("Experiment name:")
    print("\t {}".format(cfg.EXPERIMENT.name))
    print("\n" * 3)
    print("===== Experiment Configurations =====")
    print(cfg)

    trainer = PCReg_Trainer(cfg)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "train_single":
        trainer.train_epoch()
    elif args.mode == "validate":
        trainer.validate(split="valid")
    elif args.mode == "test":
        trainer.validate(split=cfg.TEST.split)
    else:
        raise ValueError("Unknown mode {}.".format(args.mode))
