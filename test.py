from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from model import PetClassifier

mean = [0.4783, 0.4459, 0.3957]
std  = [0.2254, 0.2223, 0.2240]

IMAGE_SIZE = 224

# Basically the same as the one in training, but without the test aug because obvs that ain't allowed (other than resize + tensor stuff). 
class MaskedPetDataset(Dataset):
    def __init__(self, root, split, image_transform, mask_size):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=["category", "segmentation"],
            download=True,
        )
        self.image_transform = image_transform
        self.mask_size = mask_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, (label, trimap) = self.dataset[index]
        image = self.image_transform(image)
        trimap = v2.functional.resize(trimap, [self.mask_size, self.mask_size])
        trimap_tensor = v2.functional.to_dtype(v2.functional.to_image(trimap), torch.float32, scale=False)
        mask = (trimap_tensor != 2).float()
        image = image * mask
        return image, label


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

pet_classifier = PetClassifier().to(device)
pet_classifier.load_state_dict(torch.load("model.pth"))

test_dataset = MaskedPetDataset(
    root="./data",
    split="test",
    image_transform=v2.Compose([v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),v2.ToImage(),v2.ToDtype(torch.float32, scale=True),v2.Normalize(mean=mean, std=std),]),
    mask_size=IMAGE_SIZE,
)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Size of test dataset: " + str(len(test_dataset)))

pet_classifier.eval()
nn_loss_test = nn.CrossEntropyLoss()
running_test_loss = 0.0
correct_test = 0
total_test = 0

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