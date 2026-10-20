import argparse
import json
from pathlib import Path
import torch
from models import get_model
from utils.profiler import ModelProfiler, benchmark_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "tiny_imagenet"], default="cifar10")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--output", default="results/profiles.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    variants = [("LeNet", "lenet", "none"), ("Modernized LeNet", "lenet_modern", "none"),
                ("AlexNet none", "alexnet", "none"), ("AlexNet LRN", "alexnet", "lrn"),
                ("AlexNet BN", "alexnet", "bn"), ("AlexNet BN augmented", "alexnet", "bn"),
                ("Zoo AlexNet", "zoo_alexnet", "none")]
    results = []
    measured = {}
    classes = 10 if args.dataset == "cifar10" else 200
    shape = (args.batch_size, 3, 64, 64)
    print(f"Device: {args.device}; dataset: {args.dataset}; input: {shape}; inference: FP32")
    print(f"{'Model':25} {'Params M':>10} {'MACs M':>10} {'FLOPs M':>10} {'Latency ms':>12} {'Images/s':>12}")
    for label, name, norm in variants:
        if (name, norm) in measured:
            row = dict(measured[(name, norm)])
            row["label"] = label
            row["shared_with"] = measured[(name, norm)]["label"]
            results.append(row)
            print(f"{label:25} {row['params_m']:10.4f} {row['macs_m']:10.2f} {row['flops_m']:10.2f} "
                  f"{row['latency_ms']:12.4f} {row['throughput']:12.1f} (same inference model)", flush=True)
            continue
        model = get_model(name, num_classes=classes, norm_type=norm).to(args.device)
        profiler = ModelProfiler(model, input_size=shape, device=args.device)
        latency, throughput = benchmark_inference(model, shape, args.device, reps=args.reps, warmup=20)
        row = {"label": label, "model": name, "norm": norm, "dataset": args.dataset,
               **profiler.summary(), "latency_ms": latency, "throughput": throughput,
               "reps": args.reps, "warmup": 20, "batch_size": args.batch_size, "precision": "FP32"}
        results.append(row)
        measured[(name, norm)] = row
        print(f"{label:25} {row['params_m']:10.4f} {row['macs_m']:10.2f} {row['flops_m']:10.2f} "
              f"{latency:12.4f} {throughput:12.1f}", flush=True)
        del model
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
