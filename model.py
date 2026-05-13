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
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # The layers pass from layer to the next to the next so the out of one becomes the in of the next.
        # There are five layers and I'm doubling the number of filters so that it can identify more complex features as we move deeper into the architecture.
        # I chose a kernel size of three to represent the 3x3 filter that will be applied to the images.
        # From the guest lecture on the importance of standardisation and normalisation, I have applied batch normalisation after each convolutional layer.
        # This is to ensure that the values don't get too high or too litttle
        # After applying it through each filter, I'm applying max pooling to reduce spatial dimensions, which flattens the output.
        # It's a 2x2 sliding window which essentially is taking the max val in the window, and sliding it across the image.

        # 64 to 1024 and doubling at each conv layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 37) # 37 breeds
        self.dropout = nn.Dropout(p=0.3) # It randomly sets 30% of the input units to 0 at each update during training time, which helps prevent overfitting. I set it up in anticipation of overfitting as a precautionary measure.

    def forward(self, x):
        # 64 images get fed in as a batch (which is defined per our batch size). We initially start with three filters (RGB) and we apply conv, batch norm, relu activation, and max pooling to increase the number of filters 
        # We half at each layer
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x