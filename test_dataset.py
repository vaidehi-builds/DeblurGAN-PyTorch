from datasets.gopro_dataset import GoProDataset

dataset = GoProDataset(
    root_dir=r"D:\Datasets\GOPRO",
    split="train",
    patch_size=256
)

blur, sharp = dataset[0]

print("Blur shape:", blur.shape)
print("Sharp shape:", sharp.shape)