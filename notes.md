Current Status


Dataset

- Official GoPro dataset
- 2103 train pairs
- Random 256x256 crops
- DataLoader working

Generator

- InputBlock
- DownsampleBlock x2
- ResidualBlock x9
- UpsampleBlock x2
- OutputBlock
- Full Generator verified

Input Shape:
[4,3,256,256]

Output Shape:
[4,3,256,256]