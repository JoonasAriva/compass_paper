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
from sklearn.metrics import confusion_matrix
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data.dataloader import NiftiDataModule
from src.models import build_model
from src.training.losses import build_loss
from src.training.metrics import reduce_epoch_results, single_gpu_compute_metrics

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

        self.scheduler = self._build_scheduler()

        self.global_steps = 0

        self._print(OmegaConf.to_yaml(cfg))
        self.threshold = 0.5

        if not cfg.check and self.is_main_process:
            self.run = wandb.init(project="paper", anonymous='must',
                                  settings=wandb.Settings(init_timeout=120),
                                  config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

            wandb.define_metric("f1_test", summary="max,last")
            wandb.define_metric("accuracy_test", summary="max,last")
            wandb.define_metric("f1_train", summary="max,last")
            wandb.define_metric("loss_test", summary="min,last")
            wandb.define_metric("loss_train", summary="min,last")
            wandb.define_metric("bce_loss_test", summary="min,last")
            wandb.define_metric("bce_loss_train", summary="min,last")
            wandb.define_metric("auc_roc_val", summary="max,last")
            wandb.define_metric("auc_roc_test", summary="max,last")
            wandb.define_metric("auc_roc_slice_test", summary="max,last")
            wandb.define_metric("auc_roc_slice_perscan_test", summary="max,last")

    def _build_scheduler(self):
        micro_steps = math.ceil(len(self.datamodule.train_sampler) / self.cfg.dataloader.batch_size)
        steps_per_epoch = math.ceil(micro_steps / self.cfg.grad_accumulation_steps)
        total_steps = self.cfg.epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)

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
        slice_probs = []
        slice_bags = []

        _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        accumulation_steps = self.cfg.grad_accumulation_steps if train else 1
        pending_update = False

        def _optimizer_step():
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.global_steps += 1

        data_loading_time = time.time()
        full_loop_time = time.time()
        process = psutil.Process()

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with ctx:
            for batch in tepoch:

                data_times.append(time.time() - data_loading_time)

                if self.cfg.dataloader.use_cached_features:
                    # (total_N, feat_dim) -- already instance-major, no channel dim
                    # to permute, and no padding to account for (cached tensors are
                    # always the real, compass-filtered length, so scan_end is just
                    # the total count -- same as the raw path's "no padding" case).
                    scans = batch["features"].to(self.device, non_blocking=True)
                    scan_end = scans.shape[0]
                else:
                    scans = torch.squeeze(batch["image"]).to(self.device, non_blocking=True)  # (C, total_D, H, W)
                    if self.cfg.mode == "2D":
                        scans = torch.permute(scans, (1, 0, 2, 3))  # C,D,H,W --> D,C,H,W

                    elif self.cfg.mode == "3D":
                        scans = torch.unsqueeze(torch.unsqueeze(scans[1], dim=0), dim=0)
                        # scans = torch.unsqueeze(torch.unsqueeze(scans, dim=0), dim=0)
                    if self.cfg.dataloader.batch_size == 1 and self.cfg.compass_filter == False:
                        scan_end = batch["original_depth"].item()
                    else:  # no padding
                        scan_end = scans.shape[0]
                labels = batch["class"].to(self.device, non_blocking=True).view(-1, 1).float()  # [B]->[B,1]
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
                    Y_hat = probability > self.threshold  # starts at 0.5

                    probs.append(probability.detach().cpu())
                    outputs.append(Y_hat.detach().cpu())
                    targets.append(labels.detach().cpu())

                    if self.cfg.mode == "2D":
                        individual_predictions = output["instance_scores"]

                        logit_class = (individual_predictions > 0).cpu().numpy().flatten().astype(int)

                        slice_outputs.append(logit_class)
                        slice_targets.append(batch["slice_classes"].numpy().flatten().astype(int))
                        slice_probs.append(individual_predictions.detach().float().cpu().numpy().flatten())
                        slice_bags.append(np.full(len(logit_class), step * _world + _rank, dtype=np.int64))

                        if self.cfg.check:
                            print("slice predictions: ", len(individual_predictions))
                            print("debug: ", batch["slice_classes"].shape)
                            print("slice classes: ", len(batch["slice_classes"].numpy().flatten().astype(int)))
                            assert len(logit_class) == len(batch["slice_classes"].numpy().flatten()) == len(
                                slice_bags[-1]), \
                                (len(logit_class), batch["slice_classes"].shape, len(slice_bags[-1]))

                if train:
                    backprop_time = time.time()
                    self.scaler.scale(loss["total_loss"] / accumulation_steps).backward()
                    pending_update = True
                    backprop_times.append(time.time() - backprop_time)

                    if (step + 1) % accumulation_steps == 0:
                        _optimizer_step()
                        pending_update = False

                    if len(backprop_times) % 100 == 0:
                        self._print(f"batch {len(backprop_times)} backward: {backprop_times[-1]:.3f}s "
                                    f"running avg: {np.mean(backprop_times):.3f}s")
                if step % 100 == 0:
                    self._print(f"Batch {step} CPU RAM: {process.memory_info().rss / 1e9:.2f} GB")

                results["loss"] += loss["total_loss"].item()
                # results["depth_loss"] += loss["depth_loss"].item()

                del loss, output

                step += 1

                full_loop_times.append(time.time() - full_loop_time)
                if len(full_loop_times) % 20 == 0:
                    self._print(f"batch {len(forward_times)} full loop time: {forward_times[-1]:.3f}s  "
                                f"running avg time: {np.mean(forward_times):.3f}s")
                full_loop_time = time.time()
                data_loading_time = time.time()

                if step >= 3 and self.cfg.check:
                    break

            if train and pending_update:
                _optimizer_step()

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

                results["slice_labels"] = slice_targets_flat
                results["slice_predictions"] = np.concatenate(slice_probs)
                results["slice_bag_index"] = np.concatenate(slice_bags)

        self._print(f"Rank {self.local_rank} - Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        self._print(f"Rank {self.local_rank} - Max memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        self._print(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}"),

        return results

    def _save_checkpoint(self, name, epoch, save_path=False, extra=None):
        dir_checkpoint = Path('./checkpoints/')
        if self.is_main_process:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            payload = {
                "epoch": epoch,
                "model": getattr(self.model, "module", self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "threshold": 0.5,
            }
            if extra:
                payload.update(extra)
            torch.save(payload, f"{dir_checkpoint}/{name}")
        if save_path:
            self.cfg.checkpoint_path = f"{dir_checkpoint}/{name}"

    def _load_checkpoint(self):
        ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.device)
        getattr(self.model, "module", self.model).load_state_dict(ckpt["model"])
        self.threshold = ckpt.get("threshold", 0.5)
        self._print(f"Resumed from checkpoint: {self.cfg.checkpoint_path} (threshold {self.threshold:.3f})")
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

    def _metrics(self, results):
        if torch.distributed.is_initialized():
            return reduce_epoch_results(results)
        return single_gpu_compute_metrics(self.cfg, results)

    def fit(self):

        best_test_loss = float("inf")
        best_f1 = -1
        best_auc_score = -1
        start_epoch = 0

        train_loader = self.datamodule.train_loader()
        val_loader = self.datamodule.val_loader()
        print("val loader:", val_loader)
        test_loader = self.datamodule.test_loader()

        if self.cfg.checkpoint_path:
            start_epoch = self._load_checkpoint() + 1
            self._print(f"Resuming from epoch {start_epoch}")

        train_steps_in_epoch = len(train_loader)

        process = psutil.Process()
        for epoch in range(start_epoch, self.cfg.epochs):

            sampler = self.datamodule.train_sampler
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            self._print(f"Starting epoch {epoch}")

            self._print(f"CPU RAM start: {process.memory_info().rss / 1e9:.2f} GB")

            train_results = self._run_epoch(self.model, train_loader, total_steps=train_steps_in_epoch, train=True)
            self._print(f"CPU RAM after train: {process.memory_info().rss / 1e9:.2f} GB")
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # forces glibc to return memory to OS
            self._print(f"CPU RAM after malloc trim: {process.memory_info().rss / 1e9:.2f} GB")

            # test_results = self._run_epoch(self.model, test_loader, total_steps=len(test_loader), train=False)

            epoch_results = {}
            epoch_results.update({f"{k}_train": v for k, v in self._metrics(train_results).items()})

            if val_loader is not None:
                self._print("Validating!")
                val_results = self._run_epoch(self.model, val_loader, len(val_loader), train=False)
                epoch_results.update({f"{k}_val": v for k, v in self._metrics(val_results).items()})

            if self.cfg.dataloader.eval_test_every_epoch:
                test_results = self._run_epoch(self.model, test_loader, len(test_loader), train=False)
                epoch_results.update({f"{k}_test": v for k, v in self._metrics(test_results).items()})

            if self.cfg.check:
                self._print("Model check completed")
                return

            if self.is_main_process:
                epoch_results["lr"] = self.optimizer.param_groups[0]["lr"]
                self.run.log(epoch_results)

            t = time.time()
            self._print(f"====================LOSS VALUES=========================")
            if val_loader is not None:
                self._print(
                    f"val loss: {epoch_results['loss_val']}, train loss: {epoch_results['loss_train']} at epoch {epoch}")
                if epoch_results["loss_val"] < best_test_loss:
                    best_test_loss = epoch_results["loss_val"]
                    self._print(f"Best val loss achieved {best_test_loss} at epoch {epoch}!")
                    self._save_checkpoint("best_loss.pth", epoch=epoch)

            else:
                self._print(f"train loss: {epoch_results['loss_train']} at epoch {epoch}")
            if self.cfg.loss == "bce" and val_loader is not None:
                score = epoch_results["auc_roc_val"]
                if score > best_auc_score:
                    best_auc_score = score
                    self._print(f"Best val auc roc: {best_auc_score} at epoch {epoch}")
                    self._save_checkpoint("best_val.pth", epoch=epoch, save_path=True,
                                          extra={"threshold": epoch_results["thr_val"]})

            self._save_checkpoint("current.pth", epoch=epoch, save_path=True)  # if self.cfg.loss != "bce" else False)
            self._print(f"saving checkpoint took {time.time() - t:.1f}s")

            gc.collect()
            torch.cuda.empty_cache()

    def eval(self):
        # for kits and/or Kirc
        collected = {}
        def _build_and_eval_single_dataset(dataset_name):
            self.datamodule = NiftiDataModule(self.cfg)
            test_loader = self.datamodule.test_loader()

            test_results = self._run_epoch(self.model, test_loader, total_steps=len(test_loader), train=False)
            test_metrics = self._metrics(test_results)

            epoch_results = {}
            epoch_results.update({f"{k}_test": v for k, v in test_metrics.items()})

            if self.is_main_process and getattr(self, "run", None) is not None:
                self.run.summary.update({
                    f"final/{dataset_name}/{k.replace('_test', '')}": (v.item() if hasattr(v, "item") else v)
                    for k, v in epoch_results.items()
                })
            collected[dataset_name] = epoch_results
            self._print(f"==================== {dataset_name} METRICS =========================")
            self._print(epoch_results)

            with open(f'{dataset_name}_results_.pkl', 'wb') as f:
                pickle.dump(epoch_results, f)
                self._print(f"results saved to .pkl")
            gc.collect()
            torch.cuda.empty_cache()

        self._print("Start evaluation on KITS and KIRC and TUH test")
        self.model = build_model(self.cfg).cuda()

        if not self.cfg.check:
            self._load_checkpoint()

        self.cfg.dataloader.kits = False
        self.cfg.dataloader.kirc = False
        self.cfg.dataloader.tuh = True
        self.cfg.dataloader.tuh_extra_data = False

        _build_and_eval_single_dataset("TUH")

        self.cfg.dataloader.kits = True
        self.cfg.dataloader.tuh = False

        _build_and_eval_single_dataset("KITS")

        self.cfg.dataloader.kits = False
        self.cfg.dataloader.kirc = True

        _build_and_eval_single_dataset("KIRC")

        if self.is_main_process and getattr(self, "run", None) is not None:
            self.run.summary.update({
                "final/checkpoint": str(self.cfg.checkpoint_path),
                "final/threshold": float(self.threshold),
                "final/epochs": int(self.cfg.epochs),
                "final/compass_filter": bool(self.cfg.compass_filter),
                "final/use_val_split": bool(self.cfg.dataloader.get("use_val_split", True)),
            })
            tbl = wandb.Table(columns=["dataset", "auc", "auc_slice_perscan", "f1",
                                       "recall", "specificity", "n_scans", "n_scans_slice_auc"])
            for name, r in collected.items():
                tbl.add_data(name, r.get("auc_roc_test"), r.get("auc_roc_slice_perscan_test"),
                             r.get("f1_test"), r.get("recall_test"), r.get("specificity_test"),
                             r.get("n_samples_test"), r.get("n_scans_slice_auc_test"))
            self.run.log({"final/summary_table": tbl})

        if self.cfg.check:
            self._print("Model check completed")
