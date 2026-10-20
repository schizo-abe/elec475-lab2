import json
from pathlib import Path
import torch
from torch import nn
from models import get_model


def inspect(model):
    layers, handles = [], []
    field, jump = 1, 1

    def record(name):
        def hook(module, inputs, output):
            nonlocal field, jump
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                kernel = module.kernel_size
                stride = module.stride
                kernel = kernel[0] if isinstance(kernel, tuple) else kernel
                stride = stride[0] if isinstance(stride, tuple) else stride
                field += (kernel - 1) * jump
                jump *= stride
            layers.append({"name": name, "type": type(module).__name__,
                           "output": list(output.shape)[1:],
                           "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                           "receptive_field": field, "jump": jump})
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(record(name)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64))
    for handle in handles:
        handle.remove()
    return layers


def main():
    torch.set_num_threads(2)
    output = {}
    for name in ["lenet", "lenet_modern", "alexnet", "zoo_alexnet"]:
        model = get_model(name)
        output[name] = inspect(model)
    Path("results/architectures.json").write_text(json.dumps(output, indent=2))
    print("Architecture dimensions and parameter breakdowns saved.")


if __name__ == "__main__":
    main()
