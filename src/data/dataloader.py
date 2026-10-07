import glob
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset as MonaiDataset, PersistentDataset
from monai.transforms import GridPatchd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from src.data.DDPsampler import DistributedBalancedSampler, UnevenDistributedSampler
from src.data.dataloader_utils import custom_collate, make_data_dict
from src.data.transforms import get_deterministic_transforms, get_augmentation_transforms
from src.training.compass_filter import CompassFilter


class CTDataset(TorchDataset):
    def __init__(self, data_paths, transforms, cfg, persistent_ds=None, features_dir=None):
        controls, tumors = data_paths
        self.transforms = transforms
        self.cfg = cfg
        self.patch_mode = cfg.patch_mode

        # features_dir is None -> original raw-image behaviour, unchanged.
        # features_dir is set  -> load pre-extracted tensors instead.
        self.use_cached_features = features_dir is not None
        self.features_dir = features_dir

        data_dict = make_data_dict(controls, tumors)
        self.data = data_dict

        control_labels = [[False]] * len(controls)
        self.controls = len(controls)

        tumor_labels = [[True]] * len(tumors)
        self.cases = len(tumors)

        self.img_paths = controls + tumors
        self.labels = control_labels + tumor_labels

        if cfg.compass_filter == True:
            # train_path = '/users/arivajoo/compass_paper/train_set_compass_scores_2d_slice_vol2.csv'
            # test_path = '/users/arivajoo/compass_paper/test_set_compass_scores_joined_tuh_kits_kirc_2d_slice.csv'
            train_path = "/users/arivajoo/compass_paper/train_set_compass_scores_2d_slice_okt.csv"
            test_path = "/users/arivajoo/compass_paper/test_set_compass_scores_2d_slice_okt.csv"
            self.compass_filter = CompassFilter(df_train_path=train_path, df_test_path=test_path)
        else:
            self.compass_filter = None

        # Nothing below this point (MONAI pipeline or
        # patch grid) is needed once features are pre-extracted -- all of
        # that already ran once at extraction time. img_paths/labels above
        # are still needed since the sampler and _get_cached_item both use
        # them.
        if self.use_cached_features:
            self.monai_pipeline = None
            self.grid_patch = None
            return

        self.monai_pipeline = persistent_ds if persistent_ds is not None else MonaiDataset(data=data_dict,
                                                                                           transform=transforms)

        self.grid_patch = GridPatchd(
            keys=["image", "segmentation"],
            patch_size=(1, 64, 64),
            overlap=[0.0, 0.0, 0.2, 0.2],  # no overlap in depth, 50% in H and W
            pad_mode="constant",
        )

    def _extract_patches(self, item, idx):

        patched = self.grid_patch(item)  # apply GridPatchd
        # print("debug img: ", patched["image"].shape)
        # print("debug seg: ", patched["segmentation"].shape)
        images = torch.permute(patched["image"].squeeze(2), (1, 0, 2, 3))  # (N, 3, 64, 64) -> (3, N, 64, 64)

        segs = torch.permute(patched["segmentation"].squeeze(2), (1, 0, 2, 3))

        # Background filter
        keep = (images.abs() < 0.1).float().mean(dim=(0, 2, 3)) < 0.6

        item["image"] = images[:, keep].as_tensor()
        item["segmentation"] = segs[:, keep]

        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.use_cached_features:
            return self._get_cached_item(idx)  # for frozen feature vectors

        item = self.monai_pipeline[idx]  # dim order: C,D,H,W

        if self.compass_filter:
            start_idx, end_idx = self.compass_filter.get_indexes(case_id=self.data[idx]["image"])

            if start_idx is not None and end_idx - start_idx > 1:
                item["image"] = item["image"][:, start_idx:end_idx, :, :]
                item["segmentation"] = item["segmentation"][:, start_idx:end_idx, :, :]
        else:
            item["segmentation"] = item["segmentation"][:, :item["original_depth"], :, :]
        item["scan_path"] = self.data[idx]["image"]

        if self.patch_mode:
            item = self._extract_patches(item, idx)

        if self.transforms is not None:
            item = self.transforms(item)

        seg = item["segmentation"]
        # seg shape C,D,H,W

        item["slice_classes"] = (seg == 2).any(dim=(0, 2, 3))
        item["normal_kidney_slices"] = (seg == 1).any(dim=(0, 2, 3))

        num_slices = item["slice_classes"].shape[0]
        item["bag_index"] = torch.full((num_slices,), idx, dtype=torch.long)  # each slice tagged with scan idx
        return item

    def _get_cached_item(self, idx):
        """
        Mirrors the item schema of the raw-image path as closely as possible
        (slice_classes, normal_kidney_slices, bag_index all present) -- only
        "image" is replaced by "features", and there's no "segmentation"
        since the pixel-level mask isn't meaningful once you're past a
        frozen backbone. Whatever consumes the batch downstream needs to
        branch on this same use_cached_features flag to know whether to run
        the backbone (raw mode) or use "features" directly (cached mode).

        IMPORTANT: this assumes features were extracted with compass_filter
        OFF, so the cached tensor spans the full original_depth range in
        original slice order, starting at index 0. Compass filtering is then
        applied here as a slice on the cached tensor, using the exact same
        start_idx/end_idx semantics as the raw-image path. If compass_filter
        was ON during extraction, the cache is already pre-filtered to a
        different range, and re-applying it here will slice into the wrong
        indices -- re-run extraction with compass_filter disabled first.
        """
        scan_path = self.img_paths[idx]
        stem = Path(scan_path).name.replace(".nii.gz", "").replace(".nii", "")
        cache_path = Path(self.features_dir) / f"{stem}.pt"
        # weights_only=False: these caches carry MONAI MetaTensor objects
        # (extra tracking metadata) that PyTorch 2.6+'s default weights_only
        # unpickler doesn't allowlist. Safe here since these are files you
        # generated yourself on your own scratch storage, not third-party
        # checkpoints.
        cached = torch.load(cache_path, weights_only=False)

        features = cached["features"]
        slice_classes = cached["slice_classes"]
        normal_kidney_slices = cached["normal_kidney_slices"]

        if self.compass_filter is not None:
            start_idx, end_idx = self.compass_filter.get_indexes(case_id=scan_path)
            if start_idx is not None and end_idx - start_idx > 1:
                features = features[start_idx:end_idx]
                slice_classes = slice_classes[start_idx:end_idx]
                normal_kidney_slices = normal_kidney_slices[start_idx:end_idx]

        num_instances = features.shape[0]
        return {
            "features": features,  # (N, feat_dim)
            "slice_classes": slice_classes,
            "normal_kidney_slices": normal_kidney_slices,
            "scan_path": cached["scan_path"],
            "class": cached["label"],  # same source as self.labels[idx][0]
            "bag_index": torch.full((num_instances,), idx, dtype=torch.long),
        }


class NiftiDataModule:

    def __init__(self, cfg):
        self.cfg = cfg

        self.use_cached_features = cfg.dataloader.use_cached_features

        self.train_dataset = None
        self.test_dataset = None

        train_controls, train_cases = self._collect_data_paths("train")
        test_controls, test_cases = self._collect_data_paths("test")

        use_val = (self.cfg.experiment != "compass") and self.cfg.dataloader.get("use_val_split", True)

        if not use_val:
            val_controls, val_cases = [], []
        else:
            print("Datamodule is going for the val split")
            val_ids = set(json.load(open(self.cfg.dataloader.val_split_file)))

            val_controls = [p for p in train_controls if p in val_ids]
            val_cases = [p for p in train_cases if p in val_ids]
            train_controls = [p for p in train_controls if p not in val_ids]
            train_cases = [p for p in train_cases if p not in val_ids]
            print("Val controls: ", len(val_controls), "Val cases: ", len(val_cases), "Train controls: ",
                  len(train_controls), "Train cases: ", len(train_cases))
        if self.use_cached_features:
            # No raw NIfTI / MONAI cache dir needed at all in this mode --
            # compass filtering, patch extraction, and deterministic
            # preprocessing already happened once at extraction time.
            features_dir = cfg.dataloader.features_dir
            self.train_dataset = CTDataset(
                (train_controls, train_cases), transforms=None, cfg=cfg,
                features_dir=f"{features_dir}/train",
            )
            self.test_dataset = CTDataset(
                (test_controls, test_cases), transforms=None, cfg=cfg,
                features_dir=f"{features_dir}/test",
            )
            self.val_dataset = None  # TODO for frozen feature vector experiments
        else:

            cache_dir = cfg.dataloader.cache_dir
            det_transforms = get_deterministic_transforms(cfg)

            # validate_cache(f"{cache_dir}/train")
            # validate_cache(f"{cache_dir}/test")

            train_persistent = PersistentDataset(
                data=make_data_dict(train_controls, train_cases),
                transform=det_transforms,
                cache_dir=f"{cache_dir}/train",
            )
            test_persistent = PersistentDataset(
                data=make_data_dict(test_controls, test_cases),
                transform=det_transforms,
                cache_dir=f"{cache_dir}/test",
            )
            val_persistent = PersistentDataset(data=make_data_dict(val_controls, val_cases), transform=det_transforms,
                                               cache_dir=f"{cache_dir}/train")

            self.train_dataset = CTDataset((train_controls, train_cases), get_augmentation_transforms("train"), cfg,
                                           persistent_ds=train_persistent)
            self.val_dataset = (CTDataset((val_controls, val_cases), None, cfg, persistent_ds=val_persistent)
                                if val_cases or val_controls else None)
            self.test_dataset = CTDataset((test_controls, test_cases), None, cfg, persistent_ds=test_persistent)
        if len(self.train_dataset) == 0:
            self.test_eval = True
        else:
            self.test_eval = False

        self.train_sampler, self.val_sampler, self.test_sampler = self._build_sampler()

    def _collect_data_paths(self, split: str):
        print("Split ", split)
        base = self.cfg.dataloader.base_path

        tuh_paths = [f"{base}tuh_{split}/"]
        if self.cfg.dataloader.tuh_extra_data:
            tuh_paths.append(f"{base}tuh_extra/")

        controls, tumors = [], []

        if self.cfg.dataloader.tuh:
            for path in tuh_paths:
                tuh_cases = glob.glob(f"{path}cases/images/{split}/*.nii.gz")
                tuh_controls = glob.glob(f"{path}controls/images/{split}/*.nii.gz")

                controls += tuh_controls
                tumors += tuh_cases
                print("Path: ", path, "cases: ", len(tuh_cases), "controls: ", len(tuh_controls))

        if self.cfg.dataloader.kits:
            kits = glob.glob(f"{base}data/imagesTr/{split}/kits_*.nii.gz")
            tumors += kits
            print("Kits cases: ", len(kits))

        if self.cfg.dataloader.kirc:
            kirc = glob.glob(f"{base}data/imagesTr/{split}/TCGA-*.nii.gz")
            tumors += kirc
            print("Kirc cases: ", len(kirc))

        return controls, tumors

    def _build_sampler(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if self.cfg.distributed else 1
        rank = int(os.environ.get("RANK", 0)) if self.cfg.distributed else 0

        def eval_sampler(ds):
            return UnevenDistributedSampler(ds, num_replicas=world_size, rank=rank) if self.cfg.distributed else None

        if self.test_eval:
            return None, None, eval_sampler(self.test_dataset)

        if self.cfg.experiment == "compass":  # Compass: self-supervised, labels irrelevant
            if self.cfg.distributed:
                sampler = DistributedSampler(self.train_dataset, num_replicas=world_size,
                                             rank=rank, shuffle=True, seed=self.cfg.seed)
            else:
                sampler = torch.utils.data.RandomSampler(self.train_dataset)
        else:  # classification: balanced epochs
            sampler = DistributedBalancedSampler(
                labels=[int(l[0]) for l in self.train_dataset.labels],
                num_replicas=world_size,
                rank=rank,
                seed=self.cfg.seed,
            )

        return sampler, eval_sampler(self.val_dataset) if self.val_dataset is not None else None, eval_sampler(
            self.test_dataset)

    def _make_loader(self, dataset, sampler, train: bool, shuffle: bool):
        collate_fn = partial(custom_collate, patch_mode=self.cfg.patch_mode,
                             use_cached_features=self.cfg.dataloader.use_cached_features)
        return DataLoader(
            dataset,
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.dataloader.train_workers if train else self.cfg.dataloader.val_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
            collate_fn=collate_fn,
            sampler=sampler,
            generator=torch.Generator().manual_seed(self.cfg.seed + int(os.environ.get("LOCAL_RANK", 0))),
            worker_init_fn=worker_init_fn,
        )

    def train_loader(self):
        return self._make_loader(self.train_dataset, sampler=self.train_sampler, train=True, shuffle=False)

    def test_loader(self):
        return self._make_loader(self.test_dataset, sampler=self.test_sampler, train=False,
                                 shuffle=self.cfg.notebook_eval)

    def val_loader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        return self._make_loader(self.val_dataset, sampler=self.val_sampler, train=False, shuffle=False)


import random
import monai


def worker_init_fn(worker_id):
    base = torch.initial_seed() % 2 ** 32  # already unique per worker
    np.random.seed(base)
    random.seed(base)
    monai.utils.set_determinism(seed=base)
