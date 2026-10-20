import torch
import torch.nn as nn
import torchvision.models as models

def get_torchvision_alexnet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:


    try:
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
    except AttributeError:

        model = models.alexnet(pretrained=pretrained)


    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def get_torchvision_model(model_name: str = "alexnet", num_classes: int = 10, pretrained: bool = False) -> nn.Module:


    name = model_name.lower()
    if "alexnet" in name:
        return get_torchvision_alexnet(num_classes=num_classes, pretrained=pretrained)
    elif "resnet18" in name:
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported torchvision model zoo name: '{model_name}'. Choose from 'alexnet', 'resnet18'.")
