# Please note that in my README.md, I have attached all of the links to the resources that I have opened/briefly consulted to help me with making a decision on some things, but not necessarily copied exactly in the code. 
# For the ResNet architecture, I looked at a Medium blog, which explained how the architecture works and then studied some implementations from DigitalOcean and a few other sources to understand how to implement it. 
# On the most part though, I just referred to the PyTorch docs and adapted my code at each iteration to find optimal results. 

import torch.nn as nn
import torch.nn.functional as F
import torch

# First link in the README.md helped me understand how ResNet works and then I built this implementation based off my understanding of the architecture.
# The article explained it so well, I didn't even need their code for how it works. 
# Once I created the block, I then adapted my channel sizes and layers that I originally had in my vanilla CNN to fit the ResNet architecture. 
# I increased the channel sizes from 24 to 32 as the starting point in the stem. 

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # From the guest lecture on the importance of standardisation and normalisation, I have applied batch normalisation after each convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connections
        # Basically, we need to make sure that the dimensions of the input and output match for the addition operation in the forward pass.
        # If the stride is not 1, it means that the spatial dimensions of the output will be smaller than the input, and if the number of channels changes, we also need to adjust for that.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity() # If the dimensions already match, we can just use the identity function as the shortcut to just get the input through without any changes.

    def forward(self, x):
        residual = self.shortcut(x) # This is the additional stuff which I picked up. It's the skip connection that allows the model to learn on the stuff it missed out on in the earlier layers
        activation = F.relu(self.bn1(self.conv1(x)), inplace=True)
        activation = self.bn2(self.conv2(activation))
        activation = F.relu(activation + residual, inplace=True) # This is what makes the ResNet model learn the stuff it missed out on and adjusts per the block specs.
        return activation


class BreedClassifier(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()


        # This is where my inspiration kicks in! 
        # The trimap data gives you a lot of information about the pet with regards to its edges and so on, so instead of the standard 7x7 convolutional layer in the stem,
        # , which I have seen in a lot of implementations, I chose to split it up into two separate layers and then concatenate them together. 
        # Through doing this, it does a better job of learning the earlier features of the image, which means that in turn we will build together a better understand of the overall features of the pet as the model goes deeper.
        # Therefore, we improve our chances of getting a better image accuracy. 
        # Needed to do some tweaking and tinkering with the kernel size and number of channels to optimise it, but I think that this was the best split. 
        # Started at a channel size of 24 and then adapted once I found it worked well. 
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
        
        # This is the other improvement. In the standard ResNet block, we see the block structure takes a [2,2,2,2] format but I chose to increase the third block to three layers.
        # I chose to do this because it then means that the mdoel will learn more deeper features of the image, and so we will see an increase in the overall classifictaion accuracy of the model. 
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
        self.flatten = nn.Flatten()
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