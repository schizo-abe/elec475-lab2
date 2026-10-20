import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ["lenet", "lenet_modern", "alexnet_none", "alexnet_lrn", "alexnet_bn",
            "alexnet_bn_aug", "zoo_alexnet"]
LABELS = ["Base LeNet", "Modernized LeNet", "AlexNet none", "AlexNet LRN", "AlexNet BN",
          "AlexNet BN with augmentation", "Zoo AlexNet"]
SEEDS = [42, 123, 999]


def load(dataset, variant, seed=42):
    return json.loads(Path(f"results/{dataset}_{variant}_seed{seed}.json").read_text())


def average(dataset, variant, field):
    values = []
    for seed in SEEDS:
        value = load(dataset, variant, seed)
        for part in field.split("."):
            value = value[part]
        values.append(value)
    return statistics.mean(values), statistics.stdev(values)


def curves(dataset, variants, labels, filename, metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.6, 4), squeeze=False)
    colors = ["black", "0.35", "0.6", "0.2", "0.45", "0.7", "0.1"]
    styles = ["-", "--", "-.", ":", "-", "--", "-."]
    for axis, (split, metric, title, ylabel) in zip(axes.flat, metrics):
        for index, (variant, label) in enumerate(zip(variants, labels)):
            histories = [load(dataset, variant, seed)["history"] for seed in SEEDS]
            values = np.array([[row[split][metric] for row in history] for history in histories])
            epochs = np.arange(1, values.shape[1] + 1)
            mean = values.mean(0)
            axis.plot(epochs, mean, label=label, color=colors[index], linestyle=styles[index], linewidth=1.7)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"plots/{filename}.png", dpi=200)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "black", "axes.labelcolor": "black"})
    summary = []
    for dataset in ["cifar10", "tiny_imagenet"]:
        for variant in VARIANTS + ["alexnet_bn_noaug15"]:
            row = {"dataset": dataset, "variant": variant}
            for name, field in [("accuracy", "selected_validation.accuracy"),
                                ("macro_f1", "selected_validation.macro_f1"),
                                ("train_accuracy", "selected_train.accuracy"),
                                ("gap", "generalization_gap"),
                                ("epoch_seconds", "mean_epoch_seconds")]:
                row[name + "_mean"], row[name + "_std"] = average(dataset, variant, field)
            summary.append(row)
        curves(dataset, VARIANTS, ["LeNet", "Modern", "None", "LRN", "BN", "BN augmented", "Zoo"],
               dataset + "_master_curves", [("validation", "loss", "Validation loss", "Cross entropy"),
                                             ("validation", "accuracy", "Validation accuracy", "Accuracy (%)")])
        curves(dataset, ["alexnet_bn_noaug15", "alexnet_bn_aug"], ["No augmentation", "Augmentation"],
               dataset + "_augmentation_curves", [("train", "accuracy", "Training accuracy", "Accuracy (%)"),
                                                   ("validation", "accuracy", "Validation accuracy", "Accuracy (%)")])
    for variant in ["alexnet_bn_lr0003", "alexnet_bn_lr003"]:
        row = {"dataset": "cifar10", "variant": variant}
        for name, field in [("accuracy", "selected_validation.accuracy"),
                            ("macro_f1", "selected_validation.macro_f1"),
                            ("train_accuracy", "selected_train.accuracy"),
                            ("gap", "generalization_gap"),
                            ("epoch_seconds", "mean_epoch_seconds")]:
            row[name + "_mean"], row[name + "_std"] = average("cifar10", variant, field)
        summary.append(row)
    curves("cifar10", ["lenet", "lenet_modern"], ["Base LeNet", "Modernized LeNet"], "lenet_comparison",
           [("train", "loss", "Training loss", "Cross entropy"),
            ("validation", "loss", "Validation loss", "Cross entropy")])
    norms = ["alexnet_none", "alexnet_lrn", "alexnet_bn"]
    curves("cifar10", norms, ["None", "LRN", "BatchNorm"], "normalization_loss",
           [("train", "loss", "Training loss", "Cross entropy"),
            ("validation", "loss", "Validation loss", "Cross entropy")])
    curves("cifar10", norms, ["None", "LRN", "BatchNorm"], "normalization_accuracy",
           [("train", "accuracy", "Training accuracy", "Accuracy (%)"),
            ("validation", "accuracy", "Validation accuracy", "Accuracy (%)")])
    curves("cifar10", ["alexnet_bn_lr0003", "alexnet_bn", "alexnet_bn_lr003"],
           ["Learning rate 0.0003", "Learning rate 0.001", "Learning rate 0.003"], "learning_rate_curves",
           [("train", "loss", "Training loss", "Cross entropy"),
            ("validation", "accuracy", "Validation accuracy", "Accuracy (%)")])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4))
    for axis, dataset in zip(axes, ["cifar10", "tiny_imagenet"]):
        rows = [row for row in summary if row["dataset"] == dataset and row["variant"] in VARIANTS]
        axis.bar(range(7), [row["accuracy_mean"] for row in rows],
                 yerr=[row["accuracy_std"] for row in rows], color="0.7", edgecolor="black", capsize=3)
        axis.set_xticks(range(7), ["LeNet", "Modern", "None", "LRN", "BN", "BN aug", "Zoo"], rotation=30)
        axis.set(title="CIFAR-10" if dataset == "cifar10" else "Tiny ImageNet", ylabel="Validation accuracy (%)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("plots/ablation_summary.png", dpi=200)
    plt.close(fig)
    Path("results/summary.json").write_text(json.dumps(summary, indent=2))
    with open("results/summary.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summarized {len(summary)} experiment settings with three seeds each.")


if __name__ == "__main__":
    main()
