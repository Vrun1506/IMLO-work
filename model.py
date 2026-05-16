import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # From the guest lecture on the importance of standardisation and normalisation, I have applied batch normalisation after each convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip conns
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        activation = F.relu(self.bn1(self.conv1(x)), inplace=True)
        activation = self.bn2(self.conv2(activation))
        activation = F.relu(activation + residual, inplace=True) # This is what makes the ResNet model learn the stuff it missed out on and adjusts per the block specs.
        return activation


class PetClassifier(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.stem_fine = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False) # 3x3
        self.stem_broad = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False) # 5x5
        self.stem_mix = nn.Conv2d(64, 64, kernel_size=1, bias=False) # Combine the two filters into one massive oe.
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # block strct rn = 2,2,3,2, but might increase/decrease as I see fit.
        self.stage1 = nn.Sequential(
            ResNetBlock(64, 64, stride=1),
            ResNetBlock(64, 64, stride=1),
        )

        self.stage2 = nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128, stride=1),
        )

        self.stage3 = nn.Sequential(
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 256, stride=1),
        )

        self.stage4 = nn.Sequential(
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(p=0.3) # Inrceased it to 30%
        self.fc1 = nn.Linear(512, num_classes)  # 37 breeds

    def forward(self, x):
        fine = self.stem_fine(x)
        broad = self.stem_broad(x)
        x = torch.cat([fine, broad], dim=1)
        x = F.relu(self.stem_bn(self.stem_mix(x)), inplace=True)
        x = self.stem_pool(x)  # RGB converted to 64 channels.

        x = self.stage1(x)  # Channels double, but img dims halve and keeps going down until we get one singular output.
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout1(x)
        x = self.fc1(x)
        return x