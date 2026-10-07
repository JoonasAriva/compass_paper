import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.distributed import all_reduce


def pick_threshold(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[(tpr - fpr).argmax()])


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

    if "labels" in results.keys():
        # classification task
        labels = results.pop("labels")
        predictions = results.pop("predictions")
        labels_global = gather_array(labels)
        predictions_global = gather_array(predictions)

        slice_labels = results.pop("slice_labels", None)
        slice_predictions = results.pop("slice_predictions", None)
        slice_bag_index = results.pop("slice_bag_index", None)

        has_slice = "tn_slice" in results
        reduced = all_reduce_sum_dict(results)  # tn, fp, fn, tp, n_samples, [+ slice counts]

        out = compute_confusion_metrics(reduced["tn"], reduced["fp"], reduced["fn"], reduced["tp"])

        out["thr"] = pick_threshold(labels_global, predictions_global) if len(np.unique(labels_global)) > 1 else 0.5
        out["auc_roc"] = (roc_auc_score(labels_global, predictions_global)
                          if len(np.unique(labels_global)) > 1 else float("nan"))
        out["n_samples"] = reduced["n_samples"]
        out["pos_frac"] = (reduced["tp"] + reduced["fn"]) / reduced["n_samples"]

        if has_slice:
            slice_metrics = compute_confusion_metrics(
                reduced["tn_slice"], reduced["fp_slice"], reduced["fn_slice"], reduced["tp_slice"]
            )
            for k, v in slice_metrics.items():
                out[f"{k}_slice"] = v
            out["n_samples_slice"] = reduced["n_slice_samples"]

            if slice_labels is not None:
                sl = gather_array(slice_labels)
                sp = gather_array(slice_predictions)

                out["auc_roc_slice"] = (roc_auc_score(sl, sp)
                                        if len(np.unique(sl)) > 1 else float("nan"))

                if slice_bag_index is not None:
                    bi = gather_array(slice_bag_index)
                    aucs = []
                    for b in np.unique(bi):
                        m = bi == b
                        if len(np.unique(sl[m])) > 1:
                            aucs.append(roc_auc_score(sl[m], sp[m]))
                    out["auc_roc_slice_perscan"] = float(np.mean(aucs)) if aucs else float("nan")
                    out["n_scans_slice_auc"] = len(aucs)

        for k, v in reduced.items():
            if "loss" in k:
                out[k] = v / torch.distributed.get_world_size()

    else:  # just compass training
        reduced = all_reduce_sum_dict(results)

        for k, v in reduced.items():
            if "loss" in k:
                reduced[k] = v / torch.distributed.get_world_size()
        out = reduced

    return out


def single_gpu_compute_metrics(cfg, results_dict):
    if "labels" not in results_dict:  # compass / non-classification
        return dict(results_dict)
    metrics = compute_confusion_metrics(results_dict["tn"], results_dict["fp"], results_dict["fn"],
                                        results_dict["tp"])

    metrics["auc_roc"] = (roc_auc_score(results_dict["labels"], results_dict["predictions"])
                          if len(np.unique(results_dict["labels"])) > 1 else float("nan"))
    metrics["thr"] = pick_threshold(results_dict["labels"], results_dict["predictions"]) if len(
        np.unique(results_dict["labels"])) > 1 else 0.5
    metrics["n_samples"] = results_dict["n_samples"]
    for k, v in results_dict.items():
        if "loss" in k:
            metrics[k] = v

    if cfg.mode == "2D":
        slice_metrics = compute_confusion_metrics(
            results_dict["tn_slice"], results_dict["fp_slice"],
            results_dict["fn_slice"], results_dict["tp_slice"]
        )
        for k, v in slice_metrics.items():
            metrics[f"{k}_slice"] = v
        metrics["n_samples_slice"] = results_dict["n_slice_samples"]

        slice_labels = results_dict.get("slice_labels")
        slice_preds = results_dict.get("slice_predictions")
        slice_bags = results_dict.get("slice_bag_index")

        if slice_labels is not None:
            metrics["auc_roc_slice"] = (roc_auc_score(slice_labels, slice_preds)
                                        if len(np.unique(slice_labels)) > 1 else float("nan"))

            if slice_bags is not None:
                aucs = []
                for b in np.unique(slice_bags):
                    m = slice_bags == b
                    if len(np.unique(slice_labels[m])) > 1:
                        aucs.append(roc_auc_score(slice_labels[m], slice_preds[m]))
                metrics["auc_roc_slice_perscan"] = float(np.mean(aucs)) if aucs else float("nan")
                metrics["n_scans_slice_auc"] = len(aucs)

    if cfg.loss == "bce":
        metrics["pos_frac"] = (results_dict["tp"] + results_dict["fn"]) / results_dict["n_samples"]
    return metrics
