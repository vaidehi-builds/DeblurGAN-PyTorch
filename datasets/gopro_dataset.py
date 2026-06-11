from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class GoProDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split

        self.pairs = []

        split_dir = self.root_dir / split

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            blur_dir = scene_dir / "blur"
            sharp_dir = scene_dir / "sharp"

            blur_images = sorted(blur_dir.glob("*.png"))

            for blur_path in blur_images:
                sharp_path = sharp_dir / blur_path.name

                if sharp_path.exists():
                    self.pairs.append(
                        (blur_path, sharp_path)
                    )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")

        return blur, sharp