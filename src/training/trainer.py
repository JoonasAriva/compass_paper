import ctypes
import gc
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import math
import numpy as np
import psutil
import torch
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data.dataloader import NiftiDataModule
from src.models import build_model
from src.training.losses import build_loss
from src.training.metrics import reduce_epoch_results, compute_confusion_metrics

sys.path.append('/users/arivajoo/GPAI')
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


class Trainer:
    def __init__(self, model, datamodule, cfg):

        if cfg.distributed == True:
            init_process_group(backend="nccl", timeout=timedelta(seconds=3600))
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            model.cuda()
            self.model = DDP(  # <- We need to wrap the model with DDP
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=False  # was True before
            )
        else:
            self.model = model.cuda()
            self.local_rank = 0

        self.cfg = cfg
        self.datamodule = datamodule
        self.device = torch.device("cuda")
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler()
        self.loss_function = build_loss(cfg)

        train_cases = self.datamodule.train_dataset.cases
        train_controls = self.datamodule.train_dataset.controls

        self.scheduler = self._build_scheduler(train_cases, train_controls)

        self.global_steps = 0

        self._print(OmegaConf.to_yaml(cfg))

        if not cfg.check and self.is_main_process:
            self.run = wandb.init(project="paper", anonymous='must',
                                  settings=wandb.Settings(init_timeout=120),
                                  config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

            wandb.define_metric("f1_test", summary="max")
            wandb.define_metric("accuracy_test", summary="max")
            wandb.define_metric("f1_train", summary="max")
            wandb.define_metric("loss_test", summary="min")
            wandb.define_metric("loss_train", summary="min")
            wandb.define_metric("bce_loss_test", summary="min")
            wandb.define_metric("bce_loss_train", summary="min")

    def _build_scheduler(self, train_cases, train_controls):
        steps_in_epoch = 2 * min(train_cases, train_controls)
        total_steps = self.cfg.epochs * steps_in_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps

            progress = min(
                (step - warmup_steps) / max(1, total_steps - warmup_steps),
                1.0
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )
        self.warmup_steps = warmup_steps

        return scheduler

    def _build_optimizer(self):

        backbone_decay = []
        backbone_no_decay = []
        new_decay = []
        new_no_decay = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # Determine if parameter should have NO weight decay
            # Exclude:
            # - All biases
            # - LayerNorm / GroupNorm / BatchNorm weight and bias
            if (name.endswith(".bias") or
                    "LayerNorm.weight" in name or "LayerNorm.bias" in name or
                    ".norm.weight" in name or ".norm.bias" in name or  # This catches your MyGroupNorm
                    "BatchNorm" in name):
                no_decay = True
            else:
                no_decay = False

            # Classify as backbone or new
            if name.startswith("backbone."):  # adjust prefix if needed
                if no_decay:
                    backbone_no_decay.append(p)
                else:
                    backbone_decay.append(p)
            else:
                if no_decay:
                    new_no_decay.append(p)
                else:
                    new_decay.append(p)
        # print("backbone decay: ",backbone_decay)
        # print("backbone no decay: ",backbone_no_decay)
        # print("new decay: ",new_decay)
        # print("new no decay: ",new_no_decay)
        # Optimizer with separate LR and proper weight decay
        optimizer = optim.AdamW([
            # Backbone (usually pretrained)
            {'params': backbone_decay, 'lr': 1e-4, 'weight_decay': self.cfg.weight_decay},
            {'params': backbone_no_decay, 'lr': 1e-4, 'weight_decay': 0.0},

            # New / task-specific layers
            {'params': new_decay, 'lr': 1e-4, 'weight_decay': self.cfg.weight_decay},
            {'params': new_no_decay, 'lr': 1e-4, 'weight_decay': 0.0},
        ], betas=(0.9, 0.999), eps=1e-8)

        for g in optimizer.param_groups:
            g['initial_lr'] = g['lr']
        return optimizer

    def _print(self, msg, level="info"):
        if self.is_main_process:
            getattr(logger, level)(msg)

    def _run_epoch(self, model, data_loader, total_steps, train: bool = True):

        results = defaultdict(int)  # all metrics start at zero
        step: int = 0
        total_loss: int = 0

        self.train = train

        if train:
            model.train()
            ctx = torch.set_grad_enabled(True)
            self._print("Training...")

        else:
            model.eval()
            ctx = torch.set_grad_enabled(False)
            self._print("Evaluating...")

        disable_tqdm = False if self.local_rank == 0 else True
        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=total_steps,
                      disable=disable_tqdm)

        data_times = []
        forward_times = []
        full_loop_times = []
        backprop_times = []

        # for f1 score and other classification metrics
        outputs = []
        targets = []
        probs = []
        slice_outputs = []
        slice_targets = []

        data_loading_time = time.time()
        full_loop_time = time.time()
        process = psutil.Process()
        with ctx:
            for batch in tepoch:
                self.optimizer.zero_grad(set_to_none=True)
                data_times.append(time.time() - data_loading_time)
                scans = torch.squeeze(batch["image"]).to(self.device, non_blocking=True)  # (C, total_D, H, W)
                if self.cfg.mode == "2D":
                    scans = torch.permute(scans, (1, 0, 2, 3))  # C,D,H,W --> D,C,H,W

                elif self.cfg.mode == "3D":
                    scans = torch.unsqueeze(torch.unsqueeze(scans[1], dim=0), dim=0)
                    # scans = torch.unsqueeze(torch.unsqueeze(scans, dim=0), dim=0)
                labels = batch["class"].to(self.device, non_blocking=True).view(-1, 1).float()  # [B]->[B,1]

                if self.cfg.dataloader.batch_size == 1 and self.cfg.compass_filter == False:
                    scan_end = batch["original_depth"].item()
                else:  # no padding
                    scan_end = scans.shape[0]
                bag_index = batch["bag_index"].to(self.device, non_blocking=True)

                if self.cfg.check:
                    print("scans shape: ", scans.shape, flush=True)
                    print("label shape: ", labels.shape, flush=True)
                    print("label value:", labels)
                    print("scan_end: ", scan_end, flush=True)
                    print("original_depth:", batch["original_depth"])

                forward_time = time.time()
                with torch.autocast(device_type="cuda"):

                    output = self.model(scans, scan_end=scan_end, training=train, bag_index=bag_index)

                    forward_times.append(time.time() - forward_time)

                    if len(forward_times) % 100 == 0:
                        self._print(f"batch {len(forward_times)} forward: {forward_times[-1]:.3f}s  "
                                    f"running avg: {np.mean(forward_times):.3f}s")

                    loss = self.loss_function(output["predictions"], labels=labels,
                                              z_spacing=self.cfg.dataloader.spacing[0],
                                              nth_slice=batch[
                                                  "subsample_step"] if "subsample_step" in batch.keys() else None)

                    if self.cfg.negative_instance_supervision and labels[0, 0] == 0:  # TODO extend to bs > 1

                        out = output["instance_scores"]
                        neg_labels = torch.zeros_like(out).to(self.device, non_blocking=True)

                        neg_supervision_loss = self.loss_function(out, labels=neg_labels,
                                                                  z_spacing=self.cfg.dataloader.spacing[0],
                                                                  nth_slice=batch["subsample_step"])["bce_loss"]

                        loss["total_loss"] += 0.01 * neg_supervision_loss  # 0.1 >> 0.01
                        results["neg_supervision_loss"] += neg_supervision_loss.item()

                    if self.cfg.experiment == "FocusMIL":
                        loss["total_loss"] += 0.1 * output["KL_loss"]
                        results["KL_loss"] += output["KL_loss"].item()

                if self.cfg.loss == "bce":

                    results["bce_loss"] += loss["bce_loss"].item()
                    probability = torch.sigmoid(output["predictions"])
                    Y_hat = probability > 0.5

                    probs.append(probability.detach().cpu())
                    outputs.append(Y_hat.detach().cpu())
                    targets.append(labels.detach().cpu())

                    if self.cfg.mode == "2D":
                        individual_predictions = output["instance_scores"]

                        logit_class = (individual_predictions > 0.05).cpu().numpy().flatten().astype(int)

                        slice_outputs.append(logit_class)
                        slice_targets.append(batch["slice_classes"].numpy().flatten().astype(int))

                        if self.cfg.check:
                            print("slice predictions: ", len(individual_predictions))
                            print("debug: ", batch["slice_classes"].shape)
                            print("slice classes: ", len(batch["slice_classes"].numpy().flatten().astype(int)))

                if train:
                    backprop_time = time.time()
                    self.scaler.scale(loss["total_loss"]).backward()
                    backprop_times.append(time.time() - backprop_time)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.global_steps < self.warmup_steps:
                        # Linear warmup
                        scale = (self.global_steps + 1) / float(max(1, self.warmup_steps))
                        for g in self.optimizer.param_groups:
                            g['lr'] = g['initial_lr'] * scale

                    if len(backprop_times) % 100 == 0:
                        self._print(f"batch {len(backprop_times)} backward: {backprop_times[-1]:.3f}s "
                                    f"running avg: {np.mean(backprop_times):.3f}s")
                if step % 100 == 0:
                    self._print(f"Batch {step} CPU RAM: {process.memory_info().rss / 1e9:.2f} GB")

                results["loss"] += loss["total_loss"].item()

                # results["depth_loss"] += loss["depth_loss"].item()

                del loss, output

                step += 1
                self.global_steps += 1

                full_loop_times.append(time.time() - full_loop_time)
                if len(full_loop_times) % 20 == 0:
                    self._print(f"batch {len(forward_times)} full loop time: {forward_times[-1]:.3f}s  "
                                f"running avg time: {np.mean(forward_times):.3f}s")
                full_loop_time = time.time()
                data_loading_time = time.time()

                if train and step >= total_steps:
                    break
                if step >= 3 and self.cfg.check:
                    break
        for key, value in results.items():
            results[key] = value / step

        if self.cfg.loss == "bce":
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)
            probs = np.concatenate(probs)
            # results["f1"] = f1_score(targets, outputs, average='macro')
            # if len(np.unique(targets)) > 1:
            #     results["AUC_ROC"] = roc_auc_score(targets, probs)
            # else:
            #     results["AUC_ROC"] = float("nan")
            cm = confusion_matrix(targets, outputs, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            results["tn"] = tn
            results["fp"] = fp
            results["fn"] = fn
            results["tp"] = tp
            results["n_samples"] = len(targets)
            results["labels"] = targets
            results["predictions"] = probs

            # specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            # results["specificity"] = specificity
            # results["precision"] = precision_score(targets, outputs, average='binary')
            # results["recall"] = recall_score(targets, outputs, average='binary')
            # results["accuracy"] = accuracy_score(targets, outputs)

            if self.cfg.mode == "2D":
                slice_outputs_flat = np.concatenate(slice_outputs)
                slice_targets_flat = np.concatenate(slice_targets)

                cm = confusion_matrix(slice_targets_flat, slice_outputs_flat, labels=[0, 1])
                tn_slice, fp_slice, fn_slice, tp_slice = cm.ravel()

                results["tn_slice"] = tn_slice
                results["fp_slice"] = fp_slice
                results["fn_slice"] = fn_slice
                results["tp_slice"] = tp_slice
                results["n_slice_samples"] = len(slice_outputs_flat)

                # results["slice_f1"] = f1_score(slice_targets_flat, slice_outputs_flat, average='macro', zero_division=0)
                # results["slice_precision"] = precision_score(slice_targets_flat, slice_outputs_flat, zero_division=0)
                # results["slice_recall"] = recall_score(slice_targets_flat, slice_outputs_flat, zero_division=0)
                # results["slice_accuracy"] = accuracy_score(slice_targets_flat, slice_outputs_flat)

        self._print(f"Rank {self.local_rank} - Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        self._print(f"Rank {self.local_rank} - Max memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        self._print(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}"),

        return results

    def _save_checkpoint(self, name, epoch, save_path=False):

        dir_checkpoint = Path('./checkpoints/')
        if self.is_main_process:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model": getattr(self.model, "module", self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }, f"{dir_checkpoint}/{name}")
        if save_path:
            self.cfg.checkpoint_path = f"{dir_checkpoint}/{name}"

    def _load_checkpoint(self):
        ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.device)
        getattr(self.model, "module", self.model).load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        # self.scaler.load_state_dict(ckpt["scaler"])
        self._print(f"Resumed from checkpoint: {self.cfg.checkpoint_path}")

        if self.cfg.distributed:
            torch.distributed.barrier()

        return ckpt["epoch"]

    @property
    def is_main_process(self):
        return not self.cfg.distributed or torch.distributed.get_rank() == 0

    def fit(self):

        best_test_loss = float("inf")
        best_f1 = -1
        start_epoch = 0

        train_loader = self.datamodule.train_loader()
        test_loader = self.datamodule.test_loader()

        if self.cfg.checkpoint_path:
            start_epoch = self._load_checkpoint() + 1
            self._print(f"Resuming from epoch {start_epoch}")

        gpu_count = torch.cuda.device_count()
        train_steps_in_epoch = (2 * min(self.datamodule.train_dataset.cases,
                                        self.datamodule.train_dataset.controls)) // gpu_count // self.cfg.dataloader.batch_size

        process = psutil.Process()
        for epoch in range(start_epoch, self.cfg.epochs):

            self._print(f"Starting epoch {epoch}")

            self._print(f"CPU RAM start: {process.memory_info().rss / 1e9:.2f} GB")

            train_results = self._run_epoch(self.model, train_loader, total_steps=train_steps_in_epoch, train=True)
            self._print(f"CPU RAM after train: {process.memory_info().rss / 1e9:.2f} GB")
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # forces glibc to return memory to OS
            self._print(f"CPU RAM after malloc trim: {process.memory_info().rss / 1e9:.2f} GB")

            test_results = self._run_epoch(self.model, test_loader, total_steps=len(test_loader), train=False)
            self._print(f"CPU RAM after test: {process.memory_info().rss / 1e9:.2f} GB")

            if torch.distributed.is_initialized():
                train_metrics = reduce_epoch_results(train_results)
                test_metrics = reduce_epoch_results(test_results)

            else:
                train_metrics = self.single_gpu_compute_metrics(train_results)
                test_metrics = self.single_gpu_compute_metrics(test_results)

            epoch_results = {}
            epoch_results.update({f"{k}_train": v for k, v in train_metrics.items()})
            epoch_results.update({f"{k}_test": v for k, v in test_metrics.items()})

            if self.cfg.check:
                self._print("Model check completed")
                return

            if self.is_main_process:
                self.run.log(epoch_results)

            self._print(f"====================LOSS VALUES=========================")
            self._print(
                f"train loss: {epoch_results['loss_test']}, test loss: {epoch_results['loss_test']} at epoch {epoch}")
            t = time.time()
            if epoch_results["loss_test"] < best_test_loss:
                best_test_loss = epoch_results["loss_test"]
                self._print(f"Best test loss achieved {best_test_loss} at epoch {epoch}!")
                self._save_checkpoint("best_loss.pth", epoch=epoch)
            if epoch_results["f1_test"] > best_f1:
                best_f1 = epoch_results["f1_test"]
                self._print(f"Best test f1 achieved {best_f1} at epoch {epoch}!")
                self._save_checkpoint("best_f1.pth", epoch=epoch, save_path=True)
            else:

                self._save_checkpoint("current.pth", epoch=epoch)
            self._print(f"saving checkpoint took {time.time() - t:.1f}s")

            gc.collect()
            torch.cuda.empty_cache()

    def single_gpu_compute_metrics(self, results_dict):

        metrics = compute_confusion_metrics(results_dict["tn"], results_dict["fp"], results_dict["fn"],
                                            results_dict["tp"])
        metrics["auc_roc"] = roc_auc_score(results_dict["labels"], results_dict["predictions"])
        metrics["n_samples"] = results_dict["n_samples"]
        for k, v in results_dict.items():
            if "loss" in k:
                metrics[k] = v

        if self.cfg.mode == "2D":
            slice_metrics = compute_confusion_metrics(
                results_dict["tn_slice"], results_dict["fp_slice"],
                results_dict["fn_slice"], results_dict["tp_slice"]
            )
            for k, v in slice_metrics.items():
                metrics[f"{k}_slice"] = v
            metrics["n_samples_slice"] = results_dict["n_slice_samples"]

        return metrics

    def eval(self):
        # for kits and/or Kirc
        self._print("Start evaluation on KITS and KIRC")
        self.cfg.dataloader.kits = True
        self.cfg.dataloader.kirc = False
        self.cfg.dataloader.tuh = False
        self.cfg.dataloader.tuh_extra_data = False

        self.datamodule = NiftiDataModule(self.cfg)
        test_loader = self.datamodule.test_loader()

        self.model = build_model(self.cfg).cuda()

        if not self.cfg.check:
            self._load_checkpoint()

        test_results = self._run_epoch(self.model, test_loader, total_steps=len(test_loader), train=False)

        if torch.distributed.is_initialized():
            test_metrics = reduce_epoch_results(test_results)
        else:
            test_metrics = self.single_gpu_compute_metrics(test_results)

        epoch_results = {}
        epoch_results.update({f"{k}_test": v for k, v in test_metrics.items()})

        self._print(f"==================== KITS METRICS =========================")
        self._print(epoch_results)

        with open(f'KITS_results_.pkl', 'wb') as f:
            pickle.dump(epoch_results, f)
            self._print(f"results saved to .pkl")
        gc.collect()
        torch.cuda.empty_cache()

        self.cfg.dataloader.kits = False
        self.cfg.dataloader.kirc = True

        self.datamodule = NiftiDataModule(self.cfg)
        test_loader = self.datamodule.test_loader()

        test_results = self._run_epoch(self.model, test_loader, total_steps=len(test_loader), train=False)

        if torch.distributed.is_initialized():
            test_metrics = reduce_epoch_results(test_results)
        else:
            test_metrics = self.single_gpu_compute_metrics(test_results)

        epoch_results = {}
        epoch_results.update({f"{k}_test": v for k, v in test_metrics.items()})

        self._print(f"==================== KIRC METRICS =========================")
        self._print(epoch_results)

        with open(f'KIRC_results_.pkl', 'wb') as f:
            pickle.dump(epoch_results, f)
            self._print(f"results saved to .pkl")

        if self.cfg.check:
            self._print("Model check completed")
