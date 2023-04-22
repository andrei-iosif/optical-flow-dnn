import random
import numpy as np
import torch 


def set_random_seed(seed: int):
    """ Initial seed for all RNGs used in the project.

    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """ Seed for torch DataLoader worker.

    Args:
        worker_id (int): identifier of worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)