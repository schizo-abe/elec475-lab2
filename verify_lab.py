import json
from pathlib import Path
import torch
from torch import nn
from models import get_model
from utils.dataset import get_datasets
from utils.metrics import ConfusionMatrixTracker


def main():
    torch.set_num_threads(2)
    records = []
    for classes in [10, 200]:
        for name, norm in [("lenet", "none"), ("lenet_modern", "none"),
                           ("alexnet", "none"), ("alexnet", "lrn"),
                           ("alexnet", "bn"), ("zoo_alexnet", "none")]:
            model = get_model(name, num_classes=classes, norm_type=norm)
            output = model(torch.randn(2, 3, 64, 64))
            assert output.shape == (2, classes)
            loss = nn.functional.cross_entropy(output, torch.tensor([0, classes - 1]))
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            records.append({"model": name, "norm": norm, "classes": classes,
                            "parameters": sum(p.numel() for p in model.parameters()), "backward": "passed"})
            print(records[-1], flush=True)
    train, val, classes = get_datasets("tiny_imagenet", use_augmentation=False)
    assert len(train) == 100000 and len(val) == 10000 and len(classes) == 200
    assert set(Path(path).resolve() for path, _ in train.samples).isdisjoint(
        Path(path).resolve() for path, _ in val.samples)
    assert all(val.targets.count(category) == 50 for category in range(200))
    assert train.class_to_idx == {name: index for index, name in enumerate(val.classes)}
    tracker = ConfusionMatrixTracker(2)
    tracker.update(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
    metrics = tracker.compute_metrics()
    assert abs(metrics["top1_accuracy"] - 50) < 1e-6
    assert abs(metrics["macro_f1"] - 50) < 1e-6
    Path("results/verification.json").write_text(json.dumps({"models": records,
        "tiny_validation": {"train": len(train), "validation": len(val), "classes": len(classes),
                            "validation_per_class": 50, "paths_disjoint": True},
        "metric_fixture": "passed"}, indent=2))
    print("All architecture, gradient, split and metric checks passed.")


if __name__ == "__main__":
    main()
