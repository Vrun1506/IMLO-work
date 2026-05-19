from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from model import PetClassifier

training_losses = []
training_accuracies = []
total_params = 0

# Computed on the image dataset itself and found the code for it on a PyTorch forum.
mean = [0.4783, 0.4459, 0.3957]
std  = [0.2254, 0.2223, 0.2240]

IMAGE_SIZE = 224 # Size update here. Stupidly annoying to change sizes every 5 seconds. Should've just trusted my research over trying to see if increasing the size would lead to higher acucracy.

class MaskedPetDataset(Dataset):
    # Looked at the discussion on VLE and got the idea of using the trimap stuff.
    # Applying a mask to essentially get the pet and the outline. 1 = pet, 2 = background, 3 = pet outline, and we need 1 and 3.
    # After we identify what the pixel is in the image, we then convert the pixel into a float if it's not the background and then use that as a mask to get the outline+pet data.
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
        # When we call the dataset object, it loads the images in the batch using the __getitem__ function.
        # I have inherited said method and basically tuned it so that it applies the mask to get only the desired parts. 
        # Basically, this fucntion is going to apply a mask to the image to filter ouot the noise in the backrgound 
        # So that the iomage trains better and the model focusses solely on the pet and its outline. 
        # Can't apply a colour transform to the trimap data because it will mess with the pixel values.
        # We need the pixels to figure out the regions of the image that is gonna be masked to be removed. 
        # Spatial is OK with both becuase we need to apply this to both the regular image set and the trimap data for consistency. 
        image, (label, trimap) = self.dataset[index]

        image = v2.functional.to_image(image) # Convert to tensor and normalise image data
        trimap = v2.functional.to_image(trimap) # Convert to tensor and normalise trimap data

        combined = torch.cat([image, trimap], dim=0) # Combining everything in to a single tensor so we can apply the same spatial transform
        combined = self.spatial_transform(combined)

        image = combined[:3]
        trimap = combined[3:]
        image = self.colour_transform(image) # Only applying the colour transform to the image dataset so it don't mess with the trimap data.

        # This applies the mask to the image to basically filter out the background and ensure that we only get the outline of the doggo or the cat!
        trimap_float = trimap.float() # Convert the trimap to float to make it easier to work with and apply the mask.
        mask = (torch.round(trimap_float) != 2).float() # Create a mask where pixels that are not background (2) are set to 1, and background pixels are set to 0.

        image = image * mask # Masking! Get rid of the noise. 

        return image, label


train_spatial = v2.Compose([
    v2.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    v2.RandomCrop(IMAGE_SIZE),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(15),
])

train_colour = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
    v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

training_dataset = MaskedPetDataset(
    root="./data",
    split="trainval",
    spatial_transform=train_spatial,
    colour_transform=train_colour,
    mask_size=IMAGE_SIZE,
)

# Changed back to 32 because I changed the resize so that I get finer details.
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True, drop_last=True)

print("Size of training dataset: " + str(len(training_dataset)))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

pet_classifier = PetClassifier().to(device)

for param in pet_classifier.parameters():
    total_params += param.numel()
print("Total parameters in the model: " + str(total_params))

# Add loss function.
nn_loss = nn.CrossEntropyLoss()

# Add optimiser.
optimiser = torch.optim.AdamW(pet_classifier.parameters(), lr=5e-4, weight_decay=0.02)

epoch_limit = 30
# I swapped to a OneCycleLR over just regular Cosine Annealing and found that I was getting better results.
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=5e-4, epochs=epoch_limit, steps_per_epoch=len(training_dataloader), pct_start=0.2, anneal_strategy='cos')

for epoch in range(epoch_limit):
    pet_classifier.train()
    total_epoch_loss = 0.0 # Work out average loss upon sending each batch of images to the model.

    # Acc stats at end of each epoch
    correct_train = 0
    total_train = 0

    # Load images and labels
    for images, labels in training_dataloader:
        images = images.to(device) # Send images and labels to GPU
        labels = labels.to(device)

        optimiser.zero_grad() # Zero grads before backprop

        outputs = pet_classifier(images) # Send the images through the layers
        batch_loss = nn_loss(outputs, labels) # Prediction vs label comparison to get batch loss
        batch_loss.backward() # Works out gradients of the loss
        torch.nn.utils.clip_grad_norm_(pet_classifier.parameters(), 5.0) # Stop grads getting HUGE!
        optimiser.step() # Weight update! Optimising at next batch.
        scheduler.step()

        total_epoch_loss = total_epoch_loss + batch_loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_train = total_train + labels.size(0)
        correct_train = correct_train + (predicted == labels).sum().item()

    # Stats calc
    epoch_train_loss = total_epoch_loss / len(training_dataloader)
    epoch_train_accuracy = 100.0 * correct_train / total_train
    training_losses.append(epoch_train_loss)
    training_accuracies.append(epoch_train_accuracy)

    print("\nEpoch " + str(epoch + 1) + "/" + str(epoch_limit) + "\nSummary:")
    print("Training Loss: " + str(epoch_train_loss))
    print("Training Accuracy: " + str(epoch_train_accuracy) + "%")
    print("Learning Rate: " + str(scheduler.get_last_lr()[0]))

torch.save(pet_classifier.state_dict(), "model.pth")
print("\nModel saved!")