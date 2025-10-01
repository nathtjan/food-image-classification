import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    Faithful implementation of the original AlexNet (Krizhevsky et al., 2012)
    - Original input: 227x227 RGB images (many implementations use 224x224 and it still works)
    - Uses Local Response Normalization (LRN) like the original
    - Uses grouped convolutions for conv2, conv4, conv5 (the original used 2 GPUs)
    """

    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super().__init__()

        # Features (convolutional part)
        # Conv1: 96 kernels, 11x11, stride 4, padding 2  -> output spatial shrinks a lot
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 256 kernels, 5x5, padding=2, groups=2 (split across GPUs originally)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 384 kernels, 3x3, padding=1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 384 kernels, 3x3, padding=1, groups=2
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            # Conv5: 256 kernels, 3x3, padding=1, groups=2
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Classifier (fully-connected part)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),  # assumes input spatial reduces to 6x6 (for 227 input)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # Original AlexNet used a Normal(0, 0.01) init for weights and set some biases to 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # normal with small std
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    # original chooses bias=0 for many convs, but conv2, conv4, conv5 had bias=1 in some
                    # implementations; here we follow a reasonable choice:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    # quick smoke test
    model = AlexNet(num_classes=1000)
    print(model)

    # Example input tensor: batch size 1, 3 channels, 227x227 (original AlexNet)
    x = torch.randn(1, 3, 227, 227)
    out = model(x)
    print("Output shape:", out.shape)  # expected: [1, 1000]
