from pathlib import Path

import torch
from monai.data import pad_list_data_collate


def validate_cache(cache_dir: str):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return

    corrupted = []
    for f in cache_path.glob("*.pt"):
        try:
            torch.load(f, weights_only=True, map_location="cpu")
        except Exception:
            corrupted.append(f)

    if corrupted:
        print(f"Found {len(corrupted)} corrupted cache files, deleting...")
        for f in corrupted:
            f.unlink(missing_ok=True)
    else:
        print("No corrupted cache files found.")


# def custom_collate(batch):
#     images, bag_indexes, slice_classes_list, normal_kidney_slices = [], [], [], []
#     for batch_idx, item in enumerate(batch):
#         img = item.pop("image")
#         if len(batch) > 1:
#             # Remove padding: crop to original_depth (no-op if compass filter already cropped smaller)
#             img = img[:, :item["original_depth"]]
#         actual_depth = img.shape[1]
#         images.append(img)
#         bag_indexes.append(torch.full((actual_depth,), batch_idx, dtype=torch.long))
#         item.pop("bag_index")
#         slice_classes_list.append(item.pop("slice_classes"))
#         normal_kidney_slices.append(item.pop("normal_kidney_slices"))
#
#     collated = pad_list_data_collate(batch)
#     collated["image"] = torch.cat(images, dim=1)
#     collated["bag_index"] = torch.cat(bag_indexes)
#     collated["slice_classes"] = torch.cat(slice_classes_list, dim=1)
#     collated["normal_kidney_slices"] = torch.cat(normal_kidney_slices, dim=1)
#     return collated

def custom_collate(batch, patch_mode=False, use_cached_features=False):
    if use_cached_features:
        return _collate_cached_features(batch)

    images, bag_indexes, slice_classes_list, normal_kidney_slices,segmentations = [], [], [], [], []

    for batch_idx, item in enumerate(batch):

        img = item.pop("image")
        seg = item.pop("segmentation")

        if len(batch) > 1:
            img = img[:, :item["original_depth"]]
            seg = seg[:, :item["original_depth"]]
        actual_depth = img.shape[1]
        images.append(img)
        segmentations.append(seg)
        bag_indexes.append(torch.full((actual_depth,), batch_idx, dtype=torch.long))
        item.pop("bag_index")

        slice_classes_list.append(item.pop("slice_classes"))
        normal_kidney_slices.append(item.pop("normal_kidney_slices"))

    collated = pad_list_data_collate(batch)

    collated["image"] = torch.cat(images, dim=1)
    collated["segmentation"] = torch.cat(segmentations, dim=1)

    collated["slice_classes"] = torch.cat(slice_classes_list, dim=0)

    collated["normal_kidney_slices"] = torch.cat(normal_kidney_slices, dim=0)

    collated["bag_index"] = torch.cat(bag_indexes)

    return collated


def _collate_cached_features(batch):
    """
    Same flat-concat-tagged-by-bag_index scheme as the raw-image path, just
    without the padding/truncation machinery -- cached tensors are already
    exactly the right (compass-filtered, unpadded) length per scan -- and
    without pad_list_data_collate, since nothing MONAI-specific is left in
    the item dict once features/slice_classes/normal_kidney_slices/bag_index
    are popped, keeping this path free of any MONAI dependency.
    """
    features, bag_indexes, slice_classes_list, normal_kidney_slices, scan_paths, classes = [], [], [], [], [], []

    for batch_idx, item in enumerate(batch):
        feats = item.pop("features")
        actual_len = feats.shape[0]

        features.append(feats)
        bag_indexes.append(torch.full((actual_len,), batch_idx, dtype=torch.long))
        item.pop("bag_index")  # per-scan dataset idx, discarded -- rebuilt above using batch position

        slice_classes_list.append(item.pop("slice_classes"))
        normal_kidney_slices.append(item.pop("normal_kidney_slices"))
        scan_paths.append(item.pop("scan_path"))
        classes.append(item.pop("class"))

    return {
        "features": torch.cat(features, dim=0),  # (total_N, feat_dim)
        "slice_classes": torch.cat(slice_classes_list, dim=0),
        "normal_kidney_slices": torch.cat(normal_kidney_slices, dim=0),
        "bag_index": torch.cat(bag_indexes),
        "scan_path": scan_paths,
        "class": torch.tensor(classes),  # (B,), same key as raw-image path
    }


def make_data_dict(controls, tumors):
    return (
            [{"image": p, "class": 0,
              "segmentation": p.replace("_0000.nii.gz", ".nii.gz")
            .replace("images", "labels")}
             for p in controls] +
            [{"image": p, "class": 1,
              "segmentation": p.replace("_0000.nii.gz", ".nii.gz")
            .replace("images", "labels")}
             for p in tumors]
    )
