import torch
from torch import nn


class AlexNet64(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, norm_type="bn", dropout=0.5):
        super().__init__()
        if norm_type not in ("none", "lrn", "bn"):
            raise ValueError("Normalization must be none, lrn or bn")
        self.norm_type = norm_type
        layers = []
        channels = [in_channels, 64, 192, 384, 384, 256]
        for stage in range(5):
            kernel = 5 if stage < 2 else 3
            stride = 2 if stage == 0 else 1
            layers.append(nn.Conv2d(channels[stage], channels[stage + 1], kernel,
                                    stride=stride, padding=kernel // 2))
            if norm_type == "bn":
                layers.append(nn.BatchNorm2d(channels[stage + 1]))
            layers.append(nn.ReLU(inplace=True))
            if norm_type == "lrn" and stage < 2:
                layers.append(nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0))
            if stage in (0, 1, 4):
                layers.append(nn.MaxPool2d(3, stride=2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256 * 3 * 3, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


if __name__ == "__main__":
    from torchinfo import summary
    summary(AlexNet64(), input_size=(1, 3, 64, 64))
