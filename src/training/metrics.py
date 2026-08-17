import numpy as np
import torch
from torch.distributed import all_reduce, ReduceOp
from sklearn.metrics import roc_auc_score

def gather_array(local_arr):
    """Concatenate a numpy array across all ranks. Handles uneven shard sizes."""
    world_size = torch.distributed.get_world_size()
    gathered = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_arr)
    return np.concatenate(gathered)


def all_reduce_sum_dict(d):
    """Sum scalar values across ranks in a single collective call (not an average)."""
    keys = sorted(d.keys())
    values = torch.tensor([float(d[k]) for k in keys], dtype=torch.float64).cuda()
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return dict(zip(keys, values.cpu().tolist()))


def compute_confusion_metrics(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "specificity": specificity}


def reduce_epoch_results(results):
    """Reduce one _run_epoch() output dict (train or test) into final metrics."""
    results = dict(results)  # don't mutate caller's dict

    labels = results.pop("labels")
    predictions = results.pop("predictions")
    labels_global = gather_array(labels)
    predictions_global = gather_array(predictions)

    has_slice = "tn_slice" in results
    reduced = all_reduce_sum_dict(results)  # tn, fp, fn, tp, n_samples, [+ slice counts]

    out = compute_confusion_metrics(reduced["tn"], reduced["fp"], reduced["fn"], reduced["tp"])
    out["auc_roc"] = roc_auc_score(labels_global, predictions_global)
    out["n_samples"] = reduced["n_samples"]

    for k,v in reduced.items():
        if "loss" in k:
            out[k] = v / torch.distributed.get_world_size()

    if has_slice:
        slice_metrics = compute_confusion_metrics(
            reduced["tn_slice"], reduced["fp_slice"], reduced["fn_slice"], reduced["tp_slice"]
        )
        for k, v in slice_metrics.items():
            out[f"{k}_slice"] = v
        out["n_samples_slice"] = reduced["n_slice_samples"]

    return out
