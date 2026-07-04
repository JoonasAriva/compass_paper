import torch
from monai import transforms as T
from monai.transforms import MapTransform


class SubsampleSlicesd(MapTransform):
    """
    Deterministically keep every Nth axial slice.
    Assumes shape [C, D, H, W] after EnsureChannelFirstd.
    """

    def __init__(self, keys, step: int = 3, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.default_step = step

    def __call__(self, data):
        d = dict(data)
        # Get depth from the first available key
        # for key in self.key_iterator(d):
        #     if d[key].ndim >= 4:  # safety check
        #         depth = d[key].shape[1]  # [C, D, H, W]
        #         #step = 1 if depth < 300 else self.default_step
        #         break
        #     else:
        #         # No valid key found
        #         return d
        d["subsample_step"] = self.default_step
        # Apply to all keys
        for key in self.key_iterator(d):
            d[key] = d[key][:, ::self.default_step, :, :]

        return d


class SaveShapedd(T.MapTransform):
    def __call__(self, data):
        d = dict(data)
        d["original_depth"] = d["image"].shape[1]  # (C, D, H, W) → D is index 1
        return d


class AddNeighbourSlicesd(T.MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # (1, D, H, W)
            _, D, H, W = img.shape

            x_stack = torch.empty((3, D - 2, H, W), dtype=img.dtype, device=img.device)
            x_stack[0] = img[0, :-2]  # prev
            x_stack[1] = img[0, 1:-1]  # current
            x_stack[2] = img[0, 2:]  # next

            d[key] = x_stack  # (3, D-2, H, W)
        return d


def get_deterministic_transforms(cfg):
    transforms = [
        T.LoadImaged(keys=["image", "segmentation"]),
        T.EnsureChannelFirstd(keys=["image", "segmentation"]),
        T.Orientationd(keys=["image", "segmentation"], axcodes="SPR", labels=None),
        T.Spacingd(keys=["image", "segmentation"],
                   pixdim=cfg.dataloader.spacing,
                   mode=("bilinear", "nearest")),
        T.ScaleIntensityRanged(keys=["image"],
                               a_min=cfg.dataloader.intensity_min,
                               a_max=cfg.dataloader.intensity_max,
                               b_min=0.0, b_max=1.0, clip=True)]

    if cfg.dataloader.subsample_slices:
        transforms.extend(
            [AddNeighbourSlicesd(keys=["image", "segmentation"]),
             SubsampleSlicesd(keys=["image", "segmentation"], step=3)])

    transforms += [T.CenterSpatialCropd(keys=["image", "segmentation"],
                                        roi_size=(cfg.dataloader.depth_crop,
                                                  cfg.dataloader.axial_crop,
                                                  cfg.dataloader.axial_crop)),
                   SaveShapedd(keys=["image"]),
                   T.SpatialPadd(keys=["image", "segmentation"],
                                 spatial_size=(
                                     cfg.dataloader.depth_crop, cfg.dataloader.axial_crop, cfg.dataloader.axial_crop),
                                 method="end", constant_values=0),
                   T.ToTensord(keys=["image", "segmentation"])]

    return T.Compose(transforms)


def get_augmentation_transforms(mode):
    if mode != "train":
        return None

    return T.Compose([
        T.RandFlipd(keys=["image", "segmentation"], prob=0.5, spatial_axis=1),
        T.RandRotate90d(keys=["image", "segmentation"], prob=0.5, spatial_axes=(1, 2)),
        # Mild random zoom
        # T.RandZoomd(keys=["image", "segmentation"],
        #             prob=1,
        #             min_zoom=0.9, max_zoom=1.1,
        #             mode=("bilinear", "nearest")),

        # Intensity - only on image, not segmentation
        T.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
        T.RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1)
    ])
