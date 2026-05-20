from collections import defaultdict

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet as resnet
from torchvision.models import resnet18, resnet34, resnet50, ResNet34_Weights, ResNet18_Weights, ResNet50_Weights

from src.models.model_utils import GroupNorm
from . import register

CONFIGS = {
    "resnet18": dict(block=resnet.BasicBlock, layers=[2, 2, 2, 2]),
    "resnet34": dict(block=resnet.BasicBlock, layers=[3, 4, 6, 3]),
    "resnet50": dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3]),
}

_PRETRAINED = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}

NORM_LAYERS = {
    "batch": nn.BatchNorm2d,
    "group": GroupNorm
}
import os


def load_pretrained(model, name):
    fn, weights = _PRETRAINED[name]
    sd = fn(weights=weights).state_dict()
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if int(os.environ['LOCAL_RANK']) == 0:  # for multi gpu runs to reduce the spam
        if missing:
            print(f"[pretrained] missing keys: {missing}")
        if unexpected:
            print(f"[pretrained] unexpected keys: {unexpected}")


class AttentionHead(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super(AttentionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, x):
        return self.head(x)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=128, in_dim=512):
        super(VariationalEncoder, self).__init__()
        self.fc_initial = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        hidden = self.fc_initial(x)
        mu = self.mean(hidden)
        logvar = self.logvar(hidden)

        return mu, logvar


class FocusMILClassificationHead(nn.Module):
    def __init__(self, instance_latent_dim=128):
        super(FocusMILClassificationHead, self).__init__()

        out_dim = 1

        self.fc_ins = nn.Linear(instance_latent_dim, out_dim)

    def forward(self, z_ins, bag_idx):

        loc_ins_logits = self.fc_ins(z_ins)  # [N, 1]

        # Remap bag_idx to contiguous 0..B-1
        bags, remapped_idx = bag_idx.unique(return_inverse=True)
        B = bags.shape[0]

        # Max pooling per bag: for each bag, pick the slice with highest logit
        M = torch.full((B, 1), float('-inf'), device=z_ins.device, dtype=loc_ins_logits.dtype)
        M.scatter_reduce_(0, remapped_idx.unsqueeze(1), loc_ins_logits, reduce="amax", include_self=True)

        return M, loc_ins_logits[:, 0]


@register("resnet18")
@register("resnet34")
@register("resnet50")
class ResNetCompass(nn.Module):

    def __init__(self, name: str, norm_layer: str = "batch", pretrained: str = 'imagenet', framework: str = 'compass'):
        super().__init__()
        arch_cfg = CONFIGS[name]
        norm = NORM_LAYERS[norm_layer]

        model = resnet.ResNet(arch_cfg["block"], arch_cfg["layers"], norm_layer=norm)
        if pretrained == 'imagenet':
            load_pretrained(model, name)
        elif pretrained == 'compass':
            pass  # TODO add compass trained weights here

        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.num_features = model.fc.in_features
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.framework = framework
        if framework == 'compass':
            self.classifier = nn.Linear(512, 1)
        elif framework == 'ABMIL':
            self.attention_head = AttentionHead(self.num_features, 128, 1)
            self.classifier_ab = nn.Linear(self.num_features, 1)

        elif framework == 'FocusMIL':
            self.vae = VariationalEncoder(latent_dim=128, in_dim=self.num_features)
            self.classifier_focus = FocusMILClassificationHead(instance_latent_dim=128)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            name=cfg.model,
            norm_layer=cfg.norm_layer,
            pretrained=cfg.pretrained,
            framework=cfg.experiment
        )

    def forward(self, x, scan_end, training: bool = True, bag_index=None):
        output = defaultdict(int)

        features = self.backbone(x)
        pooled_feats = self.adaptive_pooling(features)
        pooled_feats = pooled_feats[:scan_end]
        pooled_feats = pooled_feats.view(-1, 512)

        if self.framework == 'compass':
            output["predictions"] = self.classifier(pooled_feats)


        elif self.framework == 'ABMIL':
            attention = self.attention_head(pooled_feats) # (N,1)

            normed_attention = F.softmax(attention, dim=0)
            attention_aggregated_feature = (normed_attention * pooled_feats).sum(dim=0, keepdim=True)  # (1, 512)

            output["predictions"] = self.classifier_ab(attention_aggregated_feature) # (1,1)

            output["instance_scores"] = attention


        elif self.framework == 'FocusMIL':

            instance_mu, instance_logvar = self.vae(pooled_feats)
            instance_std = (instance_logvar * 0.5).exp_()

            if training:
                qzx = dist.Normal(instance_mu, instance_std)
                z_ins = qzx.rsample()  # random sampling
            else:
                z_ins = instance_mu
            bag_score, instance_scores = self.classifier_focus(z_ins, bag_index[:scan_end])
            output["predictions"] = bag_score
            output["instance_scores"] = instance_scores
            output["instance_mu"] = instance_mu
            output["instance_std"] = instance_std

            KL_loss = 0.5 * (
                    instance_mu.pow(2) + instance_std.pow(2)
                    - 2 * torch.log(instance_std + 1e-8) - 1
            ).mean()

            output["KL_loss"] = KL_loss

        return output


