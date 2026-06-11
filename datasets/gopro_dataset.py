from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class GoProDataset(Dataset):
    def __init__(self, root_dir, split="train", patch_size=256):
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size

        self.to_tensor = transforms.ToTensor()

        self.pairs = []

        split_dir = self.root_dir / split

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            blur_dir = scene_dir / "blur"
            sharp_dir = scene_dir / "sharp"

            for blur_path in sorted(blur_dir.glob("*.png")):
                sharp_path = sharp_dir / blur_path.name

                if sharp_path.exists():
                    self.pairs.append((blur_path, sharp_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")

        if self.split == "train":
            i, j, h, w = transforms.RandomCrop.get_params(
                blur,
                output_size=(self.patch_size, self.patch_size)
            )

            blur = TF.crop(blur, i, j, h, w)
            sharp = TF.crop(sharp, i, j, h, w)

        blur = self.to_tensor(blur)
        sharp = self.to_tensor(sharp)

        return blur, sharp