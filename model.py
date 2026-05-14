import torch.nn as nn
import torch.nn.functional as F
import torch


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        activation = F.relu(activation + residual, inplace=True)
        return activation


class PetClassifier(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # [2, 2, 3, 2] block strct to empahsise the more important featurea as opposed to the reugular ResNet version. 
        self.stage1 = self._make_stage(in_channels=64,  out_channels=128, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(in_channels=256, out_channels=512, num_blocks=3, stride=2)
        self.stage4 = self._make_stage(in_channels=512, out_channels=1024, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = []
        blocks.append(ResBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            blocks.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x