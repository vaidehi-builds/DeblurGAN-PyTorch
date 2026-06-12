from datasets.gopro_dataset import GoProDataset

dataset = GoProDataset(
    root_dir=r"D:\Datasets\GOPRO",
    split="train",
    patch_size=256
)

blur, sharp = dataset[0]

print("Blur min :", blur.min().item())
print("Blur max :", blur.max().item())

print("Sharp min:", sharp.min().item())
print("Sharp max:", sharp.max().item())