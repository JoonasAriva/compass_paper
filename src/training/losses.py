import torch
import torch.nn as nn
from collections import defaultdict

class DepthLoss(nn.Module):
    def __init__(self, tolerance: float = 0.0):

        super().__init__()
        self.tolerance = tolerance

    def forward(self, predictions, z_spacing, nth_slice, **kwargs):
        loss_dict = defaultdict(int)
        # predictions: (N, 1) or (N,)
        predictions = predictions.reshape(-1).float()
        N = len(predictions)
        device = predictions.device

        i, j = torch.tril_indices(N, N, offset=-1, device=device)

        pred_dist = predictions[i] - predictions[j]

        # Ground truth pairwise distances
        true_dist = (i - j).float() * (0.1 * nth_slice * z_spacing).to(device) # how many steps are we off?

        error = (pred_dist - true_dist).abs()
        if self.tolerance > 0.0:
            # Shrink error by tolerance band, then clamp negatives to 0
            # This means small deviations within tolerance are free
            slack = self.tolerance * true_dist.abs()
            error = torch.clamp(error - slack, min=0.0)

        # Lower triangle only (avoid double-counting the antisymmetric matrix)
        loss_dict["total_loss"] = error.mean()  # normalize by actual pair count
        loss_dict["depth_loss"] = error.mean()

        return loss_dict

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, predictions, labels, **kwargs):
        loss_dict = defaultdict(int)
        bce_loss = self.loss(predictions, labels)

        loss_dict["bce_loss"] = bce_loss
        loss_dict["total_loss"] = bce_loss
        return loss_dict


LOSSES = {
    "bce": ClassificationLoss,
    "depth": DepthLoss,
}


def build_loss(cfg):
    cls = LOSSES[cfg.loss]
    return cls(**cfg.loss_kwargs)  # pass any extra params from config
