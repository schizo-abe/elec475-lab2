import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ExperimentConfig:

    model_name: str = "lenet"
    norm_type: str = "bn"
    pretrained: bool = False
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (3, 64, 64)


    dataset: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    use_augmentation: bool = True


    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


    early_stopping: bool = True
    early_stopping_metric: str = "val_loss"
    patience: int = 5
    min_delta: float = 1e-4


    device: str = "cuda"


    checkpoint_dir: str = "./checkpoints"
    plot_dir: str = "./plots"
    save_model_path: str = "./checkpoints/model.pth"
    plot_file_path: str = "./plots/loss_curve.png"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
