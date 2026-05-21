# Please note that in my README.md, I have attached all of the links to the resources that I have opened/briefly consulted to help me with making a decision on some things, but not necessarily copied exactly in the code.

from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from model import BreedClassifier

# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
# To compute these custom values, I ran the code found in this forum. 
# I obviously could have used the standard ImageNet values, but I thought that using more precise values would improve my model's performance, even if it's minor. 
mean = [0.4783, 0.4459, 0.3957]
std  = [0.2254, 0.2223, 0.2240]

IMAGE_SIZE = 224

# Basically the same as the one in training, but without the training aug because obvs that ain't allowed (other than resize + tensor stuff).
class MaskedPetDataset(Dataset):
    def __init__(self, root, split, spatial_transform, colour_transform, mask_size):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=["category", "segmentation"],
            download=True,
        )
        self.spatial_transform = spatial_transform
        self.colour_transform = colour_transform
        self.mask_size = mask_size

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        image, (label, trimap) = self.dataset[index]

        image = v2.functional.to_image(image)
        trimap = v2.functional.to_image(trimap)

        combined = torch.cat([image, trimap], dim=0)
        combined = self.spatial_transform(combined)

        image = combined[:3]
        trimap = combined[3:]
        image = self.colour_transform(image)

        trimap_float = trimap.float()
        mask = (torch.round(trimap_float) != 2).float()

        image = image * mask

        return image, label


test_spatial = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)), # No other augmentations allowed in test
])

test_colour = v2.Compose([
    v2.ToDtype(torch.float32, scale=True), #No other augs allowed in test. 
    v2.Normalize(mean=mean, std=std),
])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# Load saved model from train
pet_classifier = BreedClassifier().to(device)
pet_classifier.load_state_dict(torch.load("model.pth"))

test_dataset = MaskedPetDataset(
    root="./data",
    split="test",
    spatial_transform=test_spatial,
    colour_transform=test_colour,
    mask_size=IMAGE_SIZE,
)

# Load test set
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Size of test dataset: " + str(len(test_dataset)))

pet_classifier.eval()
nn_loss_test = nn.CrossEntropyLoss()
running_test_loss = 0.0
correct_test = 0
total_test = 0

# Run through test set and observe accuracy and loss. 
with torch.no_grad(): # No gradient updates now.
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = pet_classifier(images)
        test_loss = nn_loss_test(outputs, labels)
        running_test_loss = running_test_loss + test_loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_test = total_test + labels.size(0)
        correct_test = correct_test + (predicted == labels).sum().item()

epoch_test_loss = running_test_loss / len(test_dataloader)
epoch_test_accuracy = 100.0 * correct_test / total_test

print("\nTest Summary:")
print("Test Loss: " + str(epoch_test_loss))
print("Test Accuracy: " + str(epoch_test_accuracy) + "%")