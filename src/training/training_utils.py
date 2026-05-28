import random
import torch
import numpy as np
def seed_everything(seed: int, local_rank: int = 0) -> None:
    """Seed all RNG sources deterministically for a given run seed and DDP rank."""
    # Each rank gets a unique but deterministic seed
    rank_seed = seed + local_rank

    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)  # all GPUs