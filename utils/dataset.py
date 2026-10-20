from pathlib import Path
import urllib.request
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from .transforms import get_train_transforms, get_test_transforms
from .seed import seed_worker


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_extract_tiny_imagenet(data_dir="./data"):
    target = Path(data_dir) / "tiny-imagenet-200"
    if (target / "train").is_dir() and (target / "val/val_annotations.txt").is_file():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / "tiny-imagenet-200.zip"
    if not archive.exists():
        urllib.request.urlretrieve(TINY_IMAGENET_URL, archive)
    with zipfile.ZipFile(archive) as source:
        source.extractall(target.parent)
    return str(target)


class TinyValidation(Dataset):
    def __init__(self, root, class_to_idx, transform):
        root = Path(root)
        self.transform = transform
        self.classes = list(class_to_idx)
        self.samples = []
        for line in (root / "val_annotations.txt").read_text().splitlines():
            name, category, *_ = line.split()
            self.samples.append((root / "images" / name, class_to_idx[category]))
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, target


def get_datasets(dataset="cifar10", data_dir="./data", use_augmentation=True, target_size=64):
    train_tf = get_train_transforms(dataset, use_augmentation, target_size)
    test_tf = get_test_transforms(dataset, target_size)
    if dataset == "cifar10":
        train = CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        val = CIFAR10(data_dir, train=False, download=True, transform=test_tf)
        classes = CIFAR10_CLASSES
    elif dataset == "tiny_imagenet":
        root = Path(download_and_extract_tiny_imagenet(data_dir))
        train = ImageFolder(root / "train", transform=train_tf)
        val = TinyValidation(root / "val", train.class_to_idx, test_tf)
        classes = train.classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train, val, classes


def make_loader(dataset, batch_size=128, shuffle=False, num_workers=4, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)


def get_dataloaders(dataset="cifar10", data_dir="./data", batch_size=128,
                    num_workers=4, pin_memory=True, use_augmentation=True,
                    target_size=64, seed=42):
    train, val, classes = get_datasets(dataset, data_dir, use_augmentation, target_size)
    return (make_loader(train, batch_size, True, num_workers, seed),
            make_loader(val, batch_size, False, num_workers, seed), len(classes), classes)
