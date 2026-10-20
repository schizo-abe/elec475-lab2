# ELEC 475 Lab 2

Abresham Islam Talukder, 20381857. Individual submission.

This project compares base LeNet, modernized LeNet, AlexNet with no normalization, LRN and BatchNorm, augmented AlexNet, and torchvision AlexNet on CIFAR-10 and Tiny ImageNet-200.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_lab.py
python -m utils.eda --dataset both
python prelab_verify.py
```

The course manual specifies creating the virtual environment from Conda Python. The recorded experiments used an existing isolated Ubuntu system Python 3.14.4 environment. The exact installed versions are recorded in environment_versions.txt.

## Experiments

```bash
python run_experiments.py
python prepare_results.py
python compare_models.py --dataset cifar10 --device cuda --output results/profiles_cifar10.json
python compare_models.py --dataset tiny_imagenet --device cuda --output results/profiles_tiny_imagenet.json
```

The queue contains 54 runs: seven required configurations plus a 15-epoch unaugmented BatchNorm comparison on both datasets, each with seeds 42, 123 and 999, and two additional CIFAR-10 learning rates with the same seeds. LeNet runs for 5 epochs, normalization and zoo models for 10, and the matched augmentation comparison for 15. The main learning rate is 0.001, batch size is 128 and Adam weight decay is 0.0001. Training uses CUDA bfloat16 autocast with FP32 weights when --amp is enabled. Each checkpoint is selected by validation loss. The core queue uses fixed epoch budgets; --early-stopping and --patience are available for separate experiments.

The queue skips a run only when both its result JSON and its checkpoint exist. A clone without checkpoints trains the models again. Delete a chosen result JSON to force that experiment to rerun. The commands.txt file contains every exact training command. Raw logs, complete epoch histories and summary tables are in logs/ and results/. Figures are in plots/.

The Tiny ImageNet loader uses val/val_annotations.txt and val/images, with labels aligned to the training class order. It does not evaluate on the training directory. CIFAR-10 uses its official test split as validation, following the lab template, so those reported numbers are validation benchmarks rather than untouched test estimates.

## Checkpoint evaluation

```bash
python evaluate.py --model alexnet --norm bn --dataset cifar10 --checkpoint checkpoints/alexnet_bn.pth
```

Evaluation through evaluate.py uses FP32; the training history JSON records the bfloat16-autocast evaluation used for checkpoint selection. Small prediction differences between the two precision settings are possible.

## Files

- models/ contains the four architectures.
- utils/ contains datasets, transforms, metrics, seeding, timing and plotting.
- train.py trains one experiment and writes its history and selected-checkpoint metrics.
- run_experiments.py runs the full study.
- verify_lab.py checks model shapes, gradients, Tiny ImageNet split separation and the metric implementation.
- inspect_architectures.py records layer shapes, receptive fields and parameter breakdowns.
- prepare_results.py generates the aggregate tables and comparative figures.
- compare_models.py profiles parameters and approximate operation counts and measures inference latency.

Dataset downloads, virtual environments and checkpoint files are excluded from Git. Required checkpoints are supplied in the submission archive. All run checkpoints remain in the local Ubuntu working directory. Re-running the commands recreates them.

FLOPs use twice the THOP operation count. LRN and some element-wise operations are not fully represented in that estimate. Inference timing uses synchronized FP32 forward passes; training timing includes loading, preprocessing and validation. Augmentation uses torchvision in CPU data-loader workers before transfer to CUDA.
