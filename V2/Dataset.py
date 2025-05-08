from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random

class PuzzleImageToImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, augment=False):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.augment = augment

        self.files = sorted(os.listdir(input_dir))  # assume matching filenames

        self.base_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        input_image = Image.open(os.path.join(self.input_dir, filename)).convert("RGB")
        target_image = Image.open(os.path.join(self.target_dir, filename)).convert("RGB")

        if self.augment:
            seed = random.randint(0, 2**32)  # Ensure the same augmentation is applied to both images
            random.seed(seed)
            input_image = self.augmentation_transform(input_image)
            random.seed(seed)
            target_image = self.augmentation_transform(target_image)

        return self.base_transform(input_image), self.base_transform(target_image)