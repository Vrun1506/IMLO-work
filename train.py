from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import PetClassifier

training_losses = []
training_accuracies = []

# I computed these values using the above code snippet, which I got from a PyTorch forum.
mean = [0.4783, 0.4459, 0.3957]
std = [0.2254, 0.2223, 0.2240]

training_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=v2.Compose([
        v2.RandomResizedCrop(320, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]))

# Set batch size to 32. If we wanna speed up training, increase to 64 per the documentation recommendations.
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

print("Size of training dataset: " + str(len(training_dataset)))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

pet_classifier = PetClassifier().to(device)

# Add loss function
nn_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

# Add optimiser
optimiser = torch.optim.AdamW(pet_classifier.parameters(), lr=0.0015, weight_decay=1e-3)


scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,max_lr=0.0015,epochs=30,steps_per_epoch=len(training_dataloader),pct_start=0.3,anneal_strategy='cos')

epoch_limit = 30

for epoch in range(epoch_limit):
    pet_classifier.train()
    total_epoch_loss = 0.0 # Work out average loss upon sending each batch of images to the model.

    # Acc stats at end of each epoch
    correct_train = 0
    total_train = 0

    # Load images and labels
    for images, labels in training_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        outputs = pet_classifier(images) # Send the images through the layers
        batch_loss = nn_loss(outputs, labels)
        batch_loss.backward() # Works out gradients of the loss
        optimiser.step() # Weight update! Optimising at next batch.
        scheduler.step() # Step per batch

        total_epoch_loss = total_epoch_loss + batch_loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_train = total_train + labels.size(0)
        correct_train = correct_train + (predicted == labels).sum().item()

    # Stats calc
    epoch_train_loss = total_epoch_loss / len(training_dataloader)
    epoch_train_accuracy = 100.0 * correct_train / total_train
    training_losses.append(epoch_train_loss)
    training_accuracies.append(epoch_train_accuracy)

    print("\nEpoch "+str(epoch + 1)+"/"+str(epoch_limit)+"\nSummary:")
    print("Training Loss: "+str(epoch_train_loss))
    print("Training Accuracy: "+str(epoch_train_accuracy)+"%")
    print("Learning Rate: "+str(scheduler.get_last_lr()[0]))

torch.save(pet_classifier.state_dict(), "model.pth")
print("\nModel saved!")