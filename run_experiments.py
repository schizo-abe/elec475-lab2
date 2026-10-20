import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    variants = [
        ("lenet", "lenet", "none", False, 5, 0.001),
        ("lenet_modern", "lenet_modern", "none", False, 5, 0.001),
        ("alexnet_none", "alexnet", "none", False, 10, 0.001),
        ("alexnet_lrn", "alexnet", "lrn", False, 10, 0.001),
        ("alexnet_bn", "alexnet", "bn", False, 10, 0.001),
        ("alexnet_bn_aug", "alexnet", "bn", True, 15, 0.001),
        ("zoo_alexnet", "zoo_alexnet", "none", False, 10, 0.001),
        ("alexnet_bn_noaug15", "alexnet", "bn", False, 15, 0.001)
    ]
    jobs = []
    for seed in [42, 123, 999]:
        for dataset in ["cifar10", "tiny_imagenet"]:
            for variant in variants:
                jobs.append((dataset, seed, variant))
        for name, lr in [("alexnet_bn_lr0003", 0.0003), ("alexnet_bn_lr003", 0.003)]:
            jobs.append(("cifar10", seed, (name, "alexnet", "bn", False, 10, lr)))
    for folder in ["results", "logs", "checkpoints", "plots"]:
        Path(folder).mkdir(exist_ok=True)
    commands = []
    for index, (dataset, seed, variant) in enumerate(jobs, 1):
        name, model, norm, augment, epochs, lr = variant
        key = f"{dataset}_{name}_seed{seed}"
        command = [sys.executable, "train.py", "--model", model, "--norm", norm,
                   "--dataset", dataset, "--epochs", str(epochs), "--lr", str(lr),
                   "--seed", str(seed), "--amp", "--save-model", f"checkpoints/{key}.pth",
                   "--plot-file", f"plots/{key}.png", "--results-file", f"results/{key}.json"]
        if not augment:
            command.append("--no-augment")
        commands.append(" ".join(["python", *command[1:]]))
        Path("commands.txt").write_text("\n".join(commands) + "\n")
        if Path(f"results/{key}.json").exists():
            print(f"[{index}/{len(jobs)}] Already completed {key}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] Starting {key}", flush=True)
        Path("results/progress.json").write_text(json.dumps({"current": key, "job": index,
                                                            "total": len(jobs), "started": time.time()}))
        with open(f"logs/{key}.txt", "w") as log:
            log.write("$ " + commands[-1] + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(Path(f"results/{key}.json").read_text())
        print(f"Completed {key}: accuracy={result['selected_validation']['accuracy']:.2f}; "
              f"F1={result['selected_validation']['macro_f1']:.2f}", flush=True)
    Path("results/progress.json").write_text(json.dumps({"status": "complete", "total": len(jobs)}))


if __name__ == "__main__":
    main()
