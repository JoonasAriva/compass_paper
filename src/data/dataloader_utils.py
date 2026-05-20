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


def custom_collate(batch):
    images, bag_indexes, slice_classes_list = [], [], []
    for item in batch:
        d = item["original_depth"]

        # remove padding if bs > 1
        if len(batch) > 1 or "compass_failed" in item.keys(): # if compass fails, we still need to remove padding
            images.append(item.pop("image")[:, :d])
        else:
            images.append(item.pop("image"))

        if "compass_failed" in item.keys():
            item.pop("compass_failed")

        bag_indexes.append(item.pop("bag_index"))
        slice_classes_list.append(item.pop("slice_classes"))

    collated = pad_list_data_collate(batch)
    collated["image"] = torch.cat(images, dim=1)  # (C, sum(real_D), H, W)
    collated["bag_index"] = torch.cat(bag_indexes)
    collated["slice_classes"] = torch.cat(slice_classes_list, dim=1)
    return collated


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
