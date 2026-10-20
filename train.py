import argparse
import copy
import json
from pathlib import Path
import time
import torch
from torch import nn
from models import get_model
from utils.dataset import get_dataloaders, make_loader
from utils.transforms import get_test_transforms
from utils.seed import seed_everything
from utils.profiler import Timer, ModelProfiler
from utils.visualization import plot_training_curves


def run_epoch(model, loader, device, optimizer=None, amp=False, num_classes=10):
    model.train(optimizer is not None)
    loss_sum = torch.zeros((), device=device)
    matrix = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)
    count = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        loss_sum += loss.detach() * labels.numel()
        positions = labels * num_classes + outputs.detach().argmax(1)
        matrix += torch.bincount(positions, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        count += labels.numel()
    matrix = matrix.double()
    correct = matrix.diag()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(denominator > 0, 2 * correct / denominator, 0).mean()
    return {"loss": loss_sum.item() / count, "accuracy": correct.sum().item() * 100 / count,
            "macro_f1": f1.item() * 100, "samples": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lenet", "lenet_modern", "alexnet", "zoo_alexnet"], default="lenet")
    parser.add_argument("--norm", choices=["none", "lrn", "bn"], default="bn")
    parser.add_argument("--dataset", choices=["cifar10", "tiny_imagenet"], default="cifar10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--save-model")
    parser.add_argument("--plot-file")
    parser.add_argument("--results-file", default="results/latest.json")
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(vars(args)), flush=True)
    print(f"Device: {device}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
    train_loader, val_loader, num_classes, classes = get_dataloaders(
        dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.workers, use_augmentation=not args.no_augment, seed=args.seed)
    model = get_model(args.model, num_classes=num_classes, norm_type=args.norm,
                      pretrained=args.pretrained).to(device)
    profiler = ModelProfiler(model, device=str(device))
    profiler.print_summary(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Train: {len(train_loader.dataset)}; validation: {len(val_loader.dataset)}; classes: {num_classes}", flush=True)
    print("Epoch | Train loss | Train acc | Val loss | Val acc | Macro F1 | Seconds", flush=True)
    history, best_state, best_epoch = [], None, None
    best_loss, waiting = float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        with Timer() as timer:
            train = run_epoch(model, train_loader, device, optimizer, args.amp, num_classes)
            val = run_epoch(model, val_loader, device, amp=args.amp, num_classes=num_classes)
        if not all(torch.isfinite(torch.tensor(result["loss"])) for result in [train, val]):
            raise RuntimeError("Non-finite loss")
        row = {"epoch": epoch, "train": train, "validation": val, "seconds": timer.elapsed}
        history.append(row)
        print(f"{epoch:3d} | {train['loss']:.4f} | {train['accuracy']:.2f} | {val['loss']:.4f} | "
              f"{val['accuracy']:.2f} | {val['macro_f1']:.2f} | {timer.elapsed:.2f}", flush=True)
        if val["loss"] < best_loss - args.min_delta:
            best_loss, best_epoch, waiting = val["loss"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if args.save_model:
                Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, args.save_model)
        else:
            waiting += 1
        if args.early_stopping and waiting >= args.patience:
            print("Early stopping", flush=True)
            break
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    clean_dataset = copy.copy(train_loader.dataset)
    clean_dataset.transform = get_test_transforms(args.dataset)
    clean_loader = make_loader(clean_dataset, args.batch_size, False, args.workers, args.seed)
    selected_train = run_epoch(model, clean_loader, device, amp=args.amp, num_classes=num_classes)
    selected_val = run_epoch(model, val_loader, device, amp=args.amp, num_classes=num_classes)
    result = {"config": vars(args), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "classes": classes, "profile": profiler.summary(), "history": history,
              "best_epoch": best_epoch, "selection_metric": "validation loss",
              "selected_train": selected_train, "selected_validation": selected_val,
              "generalization_gap": selected_train["accuracy"] - selected_val["accuracy"],
              "training_seconds": training_seconds,
              "mean_epoch_seconds": sum(row["seconds"] for row in history) / len(history)}
    result_path = Path(args.results_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    if args.plot_file:
        plot_training_curves([r["train"]["loss"] for r in history],
                             [r["validation"]["loss"] for r in history],
                             [r["train"]["accuracy"] for r in history],
                             [r["validation"]["accuracy"] for r in history],
                             save_path=args.plot_file, model_name=f"{args.model} {args.norm} {args.dataset}")
    print(f"Selected epoch {best_epoch}; train {selected_train['accuracy']:.2f}%; "
          f"validation {selected_val['accuracy']:.2f}%; Macro F1 {selected_val['macro_f1']:.2f}%; "
          f"gap {result['generalization_gap']:.2f} pp", flush=True)


if __name__ == "__main__":
    main()
