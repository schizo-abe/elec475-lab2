import os
import random
from typing import Sequence, Tuple, Union
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:


    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:


    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_mean_std(values: Sequence[Union[int, float]]) -> Tuple[float, float]:


    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0

    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    return mean_val, std_val

def format_mean_std(values: Sequence[Union[int, float]], unit: str = "%", decimals: int = 2) -> str:


    mean_val, std_val = calculate_mean_std(values)
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}{unit}"
