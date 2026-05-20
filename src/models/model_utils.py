import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(GroupNorm, self).__init__()
        num_groups = max(1, num_channels // 8)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x
