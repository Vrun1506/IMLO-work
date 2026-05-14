from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset
import torch

IMAGE_SIZE = 320

mean = [0.4783, 0.4459, 0.3957]
std = [0.2254, 0.2223, 0.2240]

class MaskedPetDataset(Dataset):
    # Looked at the discussion on VLE and got the idea of using the trimap stuff. 
    # Requires a channel update (that's why we went from 3 to 4 channels, but it's fine because it improved our accuracy by like 5%). 
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

    def __getitem__(self, idx):
        image, (label, trimap) = self.dataset[idx]

        image = v2.functional.to_image(image)
        trimap = v2.functional.to_image(trimap)

        combined = torch.cat([image, trimap], dim=0)
        combined = self.spatial_transform(combined)

        image = combined[:3]
        trimap = combined[3:]
        image = self.colour_transform(image)

        # This applies the mask to the image to basically filter out the background and ensure that we only get the outline of the doggo or the cat!
        trimap_float = trimap.float()
        mask = (torch.round(trimap_float) != 2).float()

        image = image * mask

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