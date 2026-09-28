from operator import itemgetter
from typing import Iterator, Optional
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DistributedBalancedSampler(torch.utils.data.Sampler):
    """Each epoch: all positives + an equal number of random negatives, shuffled, split across ranks."""
    def __init__(self, labels, num_replicas, rank, seed=0):
        labels = torch.as_tensor(labels)
        self.pos, self.neg = torch.where(labels == 1)[0], torch.where(labels == 0)[0]
        self.n = min(len(self.pos), len(self.neg))
        self.num_replicas, self.rank, self.seed, self.epoch = num_replicas, rank, seed, 0
        self.num_samples = (2 * self.n) // num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)  # same on every rank
        pos = self.pos[torch.randperm(len(self.pos), generator=g)[: self.n]]
        neg = self.neg[torch.randperm(len(self.neg), generator=g)[: self.n]]
        idx = torch.cat([pos, neg])[torch.randperm(2 * self.n, generator=g)]
        idx = idx[: self.num_samples * self.num_replicas]
        return iter(idx[self.rank :: self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples


class UnevenDistributedSampler(torch.utils.data.Sampler):
    """Split a dataset across ranks with no padding: every sample seen exactly once."""
    def __init__(self, dataset, num_replicas, rank):
        self.indices = list(range(len(dataset)))[rank::num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)