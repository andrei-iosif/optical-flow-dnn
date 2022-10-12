import os
import random

import numpy as np
import torch


def check_device(gpu_number):
    if torch.cuda.is_available():
        dev = torch.device("cuda", int(gpu_number))
        torch.cuda.current_device()
    else:
        dev = torch.device("cpu")

    torch.cuda.empty_cache()

    print(f"Using device: {dev}")
    return dev


def set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
