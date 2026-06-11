from datasets.gopro_dataset import GoProDataset

dataset = GoProDataset(
    root_dir=r"D:\Datasets\GOPRO",
    split="train"
)

for idx in [0, 100, 500, 1000, 2000]:
    blur, sharp = dataset[idx]

    print(f"Pair {idx}")
    blur.show()
    sharp.show()

    input("Press Enter...")