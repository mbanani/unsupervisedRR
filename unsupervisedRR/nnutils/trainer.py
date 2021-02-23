"""
A simple trainer class method that allows for easily transferring major
utilities/boiler plate code.
"""
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..datasets import build_loader
from ..models import build_model
from ..utils.io import makedir


def none_grad(model):
    for p in model.parameters():
        p.grad = None


class BaseEngine:
    """
    Basic engine class that can be extended to be a trainer or evaluater.
    Captures default settings for building losses/metrics/optimizers.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # assuming single gpu jobs for now
        self.device = torch.cuda.current_device()
        self.model = build_model(cfg.MODEL).to(self.device)


class BasicTrainer(BaseEngine):
    def __init__(self, cfg):
        super(BasicTrainer, self).__init__(cfg)
        # For reproducibility -
        # refer https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(cfg.SYSTEM.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.SYSTEM.RANDOM_SEED)
        np.random.seed(cfg.SYSTEM.RANDOM_SEED)

        # Define dataset loaders
        # Overfitting steps instead of epochs
        if cfg.DATASET.overfit:
            self.train_loader = build_loader(cfg.DATASET, split="train", overfit=100)
            self.valid_loader = build_loader(cfg.DATASET, split="train", overfit=1)
        else:
            self.train_loader = build_loader(cfg.DATASET, split="train")
            self.valid_loader = build_loader(cfg.DATASET, split="valid")

        # get a single instance; just for debugging purposes
        self.train_loader.dataset.__getitem__(0)
        self.valid_loader.dataset.__getitem__(0)

        # Define some useful training parameters
        self.epoch = 0
        self.step = 0
        self.eval_step = 100 if cfg.DATASET.overfit else cfg.TRAIN.eval_step
        self.num_epochs = cfg.TRAIN.num_epochs
        self.vis_step = cfg.TRAIN.vis_step
        self.best_loss = 1e9  # very large number
        self.curr_loss = 1e8  # slightly smaller very large number
        self.training = True

        # Define Solvers
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Restore from checkpoint if a path is provided
        if cfg.EXPERIMENT.checkpoint != "":
            self.restore_checkpoint(cfg.EXPERIMENT.checkpoint)
            self.full_exp_name = "resumed_{}_{:}".format(
                cfg.EXPERIMENT.name, time.strftime("%m%d-%H%M"),
            )
        else:
            if cfg.EXPERIMENT.name == "DEBUG":
                self.full_exp_name = cfg.EXPERIMENT.name
            else:
                self.full_exp_name = "{}_{:}".format(
                    cfg.EXPERIMENT.name, time.strftime("%m%d-%H%M"),
                )
            print("Full experiment name: {}".format(self.full_exp_name))

        # Define logger
        log_dir = os.path.join(cfg.PATHS.tensorboard_dir, self.full_exp_name)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Define experimental dir for checkpoints
        exp_dir = os.path.join(cfg.PATHS.experiments_dir, self.full_exp_name)
        makedir(exp_dir, replace_existing=True)
        self.experiment_dir = exp_dir

    def save_checkpoint(self):
        if self.step == 0:
            return
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()

        # checkpoint_dict
        checkpoint = {
            "model": model_state,
            "optim": optim_state,
            "curr_loss": self.curr_loss,
            "best_loss": self.best_loss,
            "epoch": self.epoch,
            "step": self.step,
            "cfg": self.cfg,
        }

        name = "checkpoint@e{:04d}s{:07d}.pkl".format(self.epoch, self.step)
        path = os.path.join(self.experiment_dir, name)

        print("Saved a checkpoint {}".format(name))
        torch.save(checkpoint, path)

        if self.curr_loss == self.best_loss:
            # Not clear if best loss is best accuracy, but whatever
            print("Best model so far, with a loss of {}".format(self.best_loss))
            path = os.path.join(self.experiment_dir, "best_loss.pkl")
            torch.save(checkpoint, path)

        # return model to state
        self.model.to(self.device)

    def restore_checkpoint(self, checkpoint_path):
        print("Restoring checkpoint {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.curr_loss = checkpoint["curr_loss"]
        self.best_loss = checkpoint["best_loss"]

        # update model params
        old_dict = checkpoint["model"]
        model_dict = {}
        for k in old_dict:
            if "module" == k[0:6]:
                model_dict[k[7:]] = old_dict[k]
            else:
                model_dict[k] = old_dict[k]

        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        # update optim params
        self.optimizer.load_state_dict(checkpoint["optim"])

    def build_optimizer(self, network=None):
        # Currently just taking all the model parameters
        cfg = self.cfg.TRAIN
        if network is None:
            network = self.model
        params = network.parameters()

        # Define optimizer
        if cfg.optimizer == "SGD":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=False,
            )
        elif cfg.optimizer == "Adam":
            return torch.optim.Adam(
                params, lr=cfg.lr, eps=1e-4, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "RMS":
            return torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            raise ValueError()

    def build_scheduler(self):
        cfg = self.cfg.TRAIN

        if cfg.scheduler == "constant":
            # setting gamma to 1 means constant LR
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=1
            )
        else:
            raise ValueError("Scheduler {} not recognized".format(cfg.scheduler))

        return scheduler

    def _prepare_metric(self, x):
        x = x.detach().cpu()
        return x

    def _update_metrics(self, epoch_metrics, batch_metrics):
        # Update metrics
        for k in batch_metrics:
            b_metrics = self._prepare_metric(batch_metrics[k])

            if k in epoch_metrics:
                epoch_metrics[k] = torch.cat((epoch_metrics[k], b_metrics), dim=0,)
            else:
                epoch_metrics[k] = b_metrics
        return epoch_metrics

    def log_dict(self, log_dict, header, split):
        if self.logger is None:
            return
        for key in log_dict:
            val = log_dict[key]
            if "torch" in str(type(val)):
                val = val.mean().item()

            if split is None:
                tab = header
            else:
                key_parts = key.split("_")
                if np.isscalar(key_parts[-1]):
                    tab = "_".join(key_parts[:-1])
                else:
                    tab = key
                key = key + "_" + split
            if np.isscalar(val) or len(val.shape) == 1:
                self.logger.add_scalar("{}/{}".format(tab, key), val, self.step)
            elif len(val.shape) in [2, 3]:
                pass
            else:
                raise ValueError("Cannot log {} on tensorboard".format(val))

    def forward_preprocess(self):
        pass

    def calculate_norm_dict(self):
        grad_fn = torch.nn.utils.clip_grad_norm_
        grad_norm = grad_fn(self.model.parameters(), 10 ** 20)
        return {"full_model": grad_norm.item()}

    def train_epoch(self):
        epoch_loss = 0
        e_metrics = {}
        norm_dict = {}
        time_dict = {}
        d_metrics = {}
        self.model.train()

        # Setup tqdm loader -- no description for now
        disable_tqdm = not self.cfg.SYSTEM.TQDM
        t_loader = tqdm(self.train_loader, disable=disable_tqdm, dynamic_ncols=True,)

        # Begin training
        before_load = time.time()
        for i, batch in enumerate(t_loader):
            self.forward_preprocess()
            none_grad(self.model)
            after_load = time.time()
            self.step += 1

            # Forward pass
            b_loss, b_metrics, b_outputs = self.forward_batch(batch)

            after_forward = time.time()

            # Backward pass
            b_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            after_backward = time.time()
            norm_dict = self.calculate_norm_dict()

            # calculate times
            load_time = after_load - before_load
            fore_time = after_forward - after_load
            back_time = after_backward - after_forward
            total_time = load_time + fore_time + back_time

            time_dict["total_time"] = total_time
            time_dict["data_ratio"] = load_time / total_time
            time_dict["fore_ratio"] = fore_time / total_time
            time_dict["back_ratio"] = back_time / total_time

            # Log results
            e_metrics = self._update_metrics(e_metrics, b_metrics)
            e_metrics = self._update_metrics(e_metrics, d_metrics)
            epoch_loss += b_loss.item()

            if (i % 10) == 0:
                self.log_dict(b_metrics, "metrics", "train")
                self.log_dict(d_metrics, "metrics", "train")
                self.log_dict(norm_dict, "grad_norm", None)
                self.log_dict(time_dict, "time", None)

            # Validate and save checkpoint based on step?
            if (self.step % self.eval_step) == 0:
                none_grad(self.model)
                self.validate()
                self.model.train()

            # reset timer
            before_load = time.time()

        # Log results
        # Scale down calculate metrics
        epoch_loss /= len(self.train_loader)
        print("Training Metrics: ")
        metric_keys = list(e_metrics.keys())
        metric_keys.sort()
        for m in metric_keys:
            print("    {:25}:   {:10.5f}".format(m, e_metrics[m].mean().item()))

    def validate(self, split="valid"):
        # choose loader:
        if split == "valid":
            data_loader = self.valid_loader
        else:
            raise ValueError()

        v_loss = torch.zeros(1).to(self.device)
        v_metrics = {}
        self.training = False
        self.model.eval()

        # Setup tqdm loader -- no description for now
        disable_tqdm = not self.cfg.SYSTEM.TQDM
        tqdm_loader = tqdm(data_loader, disable=disable_tqdm, dynamic_ncols=True,)

        for i, batch in enumerate(tqdm_loader):
            # Forward pass
            b_loss, b_metrics, b_outputs = self.forward_batch(batch)

            # Log results
            v_metrics = self._update_metrics(v_metrics, b_metrics)
            v_loss += b_loss.detach()
            self.logger.flush()

        # Scale down calculate metrics
        v_loss /= len(data_loader)

        for k in v_metrics:
            # very hacky -- move to gpu to sync; mean first to limit gpu memory
            v_metrics[k] = v_metrics[k].mean().to(self.device)

        # Log results
        self.log_dict(v_metrics, "metrics", split)
        print("Validation after {} epochs".format(self.epoch))
        print("  {} Metrics: ".format(split))
        metric_keys = list(v_metrics.keys())
        metric_keys.sort()
        for m in metric_keys:
            print("    {:25}:   {:10.5f}".format(m, v_metrics[m].mean().item()))

        self.curr_loss = v_loss
        # Update best loss
        if v_loss < self.best_loss:
            self.best_loss = v_loss

        # save model
        self.save_checkpoint()

        # Restore training setup
        self.training = True
        return v_metrics

    def train(self):
        self.validate()
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()
            self.epoch += 1
