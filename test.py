from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model import PetClassifier

# I computed these values using the above code snippet, which I got from a PyTorch forum.
mean = [0.4783, 0.4459, 0.3957]
std = [0.2254, 0.2223, 0.2240]


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

pet_classifier = PetClassifier().to(device)
pet_classifier.load_state_dict(torch.load("model.pth", map_location=device))
print("Model loaded from model.pth")

test_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=True,
    transform=v2.Compose([
        v2.Resize((320, 320)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ]))

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Size of test dataset: " + str(len(test_dataset)))

pet_classifier.eval()
nn_loss = nn.CrossEntropyLoss()
running_test_loss = 0.0
correct_test = 0
total_test = 0

# Basically the same process as above but for the validation part
with torch.no_grad(): # We don't wanna check or update the gradients or the weights here, so no need for backprop
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = pet_classifier(images)
        batch_loss = nn_loss(outputs, labels)
        running_test_loss = running_test_loss + batch_loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_test = total_test + labels.size(0)
        correct_test = correct_test + (predicted == labels).sum().item()

# Stats calc
test_loss = running_test_loss / len(test_dataloader)
test_accuracy = 100.0 * correct_test / total_test

print("\nTest Summary:")
print("Test Loss: " + str(test_loss))
print("Test Accuracy: " + str(test_accuracy) + "%")