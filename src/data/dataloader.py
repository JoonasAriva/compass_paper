import glob
import os

import numpy as np
import torch
from monai.data import Dataset as MonaiDataset, PersistentDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from src.data.DDPsampler import DistributedSamplerWrapper
from src.data.dataloader_utils import custom_collate, make_data_dict
from src.data.transforms import get_deterministic_transforms, get_augmentation_transforms
from src.training.compass_filter import CompassFilter


class CTDataset(TorchDataset):
    def __init__(self, data_paths, transforms, cfg, persistent_ds=None):
        controls, tumors = data_paths
        self.transforms = transforms
        self.cfg = cfg

        data_dict = make_data_dict(controls, tumors)
        self.data = data_dict

        control_labels = [[False]] * len(controls)
        self.controls = len(controls)

        tumor_labels = [[True]] * len(tumors)
        self.cases = len(tumors)

        self.img_paths = controls + tumors
        self.labels = control_labels + tumor_labels

        self.monai_pipeline = persistent_ds if persistent_ds is not None else MonaiDataset(data=data_dict,
                                                                                           transform=transforms)

        if cfg.compass_filter == True:
            train_path = '/users/arivajoo/compass_paper/train_set_compass_scores_2d_slice.csv'
            test_path = '/users/arivajoo/compass_paper/test_set_compass_scores_2d_slice.csv'
            self.compass_filter = CompassFilter(df_train_path=train_path, df_test_path=test_path)
        else:
            self.compass_filter = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.monai_pipeline[idx]  # dim order: C,D,H,W
        if self.transforms is not None:
           item = self.transforms(item)

        if self.compass_filter:
            start_idx, end_idx = self.compass_filter.get_indexes(case_id=self.data[idx]["image"])

            if start_idx is not None and end_idx > start_idx:
                item["image"] = item["image"][:, start_idx:end_idx, :, :]
                item["segmentation"] = item["segmentation"][:, start_idx:end_idx, :, :]

        seg = item["segmentation"]
        # seg shape 1,D, H,W
        item["slice_classes"] = (seg == 2).any(dim=(2, 3))
        item["normal_kidney_slices"] = (seg == 1).any(dim=(2, 3))
        item["scan_path"] = self.data[idx]["image"]
        num_slices = item["image"].shape[1]
        item["bag_index"] = torch.full((num_slices,), idx, dtype=torch.long)  # each slice tagged with scan idx

        return item


class NiftiDataModule:

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dataset = None
        self.test_dataset = None

        train_controls, train_cases = self._collect_data_paths("train")
        test_controls, test_cases = self._collect_data_paths("test")

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

        self.train_dataset = CTDataset((train_controls, train_cases), get_augmentation_transforms("train"), cfg,
                                       persistent_ds=train_persistent)
        self.test_dataset = CTDataset((test_controls, test_cases), None, cfg, persistent_ds=test_persistent)
        self.train_sampler, self.test_sampler = self._build_sampler()

    def _collect_data_paths(self, split: str):
        base = self.cfg.dataloader.base_path

        tuh_paths = [f"{base}tuh_train/", f"{base}tuh_test/"]
        if split == "test" or self.cfg.dataloader.tuh_extra_data:
            tuh_paths.append(f"{base}tuh_extra/")

        controls, tumors = [], []
        for path in tuh_paths:
            controls += glob.glob(f"{path}controls/images/{split}/*.nii.gz")
            tumors += glob.glob(f"{path}cases/images/{split}/*.nii.gz")

        if not self.cfg.dataloader.tuh_only:
            tumors += glob.glob(f"{base}data/imagesTr/{split}/*.nii.gz")
        return controls, tumors

    def _build_sampler(self):

        class_sample_count = [self.train_dataset.controls, self.train_dataset.cases]
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weight = np.array([weights[int(t[0])] for t in self.train_dataset.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                 replacement=False)

        if self.cfg.distributed:
            sampler = DistributedSamplerWrapper(sampler=sampler, num_replicas=int(torch.cuda.device_count()),
                                                rank=int(os.environ["LOCAL_RANK"]), shuffle=True)
            sampler_test = DistributedSampler(self.test_dataset, num_replicas=int(torch.cuda.device_count()),
                                              rank=int(os.environ["LOCAL_RANK"]),
                                              shuffle=True)
        else:
            sampler_test = None

        return sampler, sampler_test

    def _make_loader(self, dataset, sampler, train: bool, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.dataloader.train_workers if train else self.cfg.dataloader.val_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
            sampler=sampler,
            collate_fn=custom_collate,
            generator=torch.Generator().manual_seed(self.cfg.seed)
        )

    def train_loader(self):
        return self._make_loader(self.train_dataset, sampler=self.train_sampler, train=True, shuffle=False)

    def test_loader(self):
        return self._make_loader(self.test_dataset, sampler=self.test_sampler, train=False, shuffle=self.cfg.notebook_eval)
