from torch.utils.data import DataLoader
from datasets.gopro_dataset import GoProDataset

dataset = GoProDataset(
    root_dir=r"D:\Datasets\GOPRO",
    split="train",
    patch_size=256
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

blur, sharp = next(iter(loader))

print("Blur batch shape :", blur.shape)
print("Sharp batch shape:", sharp.shape)