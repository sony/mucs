import sys, os, argparse
import torch, torchvision
from tqdm import tqdm
import math

from data_utils import CelebA

"""
Datasets are stored in a folder with a metadata file (see below) and a
subfolder named "pt", nested by the last two indices...
"""

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default=None, required=True)
parser.add_argument("--dest_dir", type=str, default=None, required=True)
parser.add_argument("--images_folder", type=str, default="pt_files", required=False)
parser.add_argument("--image_size", type=int, default=64, required=False)
parser.add_argument("--max_sample", type=int, default=50000, required=False)
parser.add_argument("--num_examples", type=int, default=100, required=False)
args, args_extra = parser.parse_known_args()

args.dest_dir = os.path.abspath(args.dest_dir)

# Create folder structure
if os.path.exists(args.dest_dir):
    print("Clean destination folder...")
    os.system("rm -rf " + args.dest_dir)
os.makedirs(args.dest_dir)

# Get datasets
print("Extracting data...")
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
    ]
)
ds_train = CelebA(
    root=args.source_dir,
    split="train",
    transform=transforms,
)
ds_valid = CelebA(
    root=args.source_dir,
    split="valid",
    transform=transforms,
)
ds_test = CelebA(
    root=args.source_dir,
    split="test",
    transform=transforms,
)
strlen_idx = len(str(len(ds_train) + len(ds_valid) + len(ds_test) - 1))

# Create metadata
metadata = {
    "dataset_name": "celeba",
    "image_size": None,  # Computed after loading the data
    "image_mean": None,  # Computed after loading the data
    "image_std": None,  # Computed after loading the data
    "label_dim": 1,  # force unconditional
    "num_samples": len(ds_train) + len(ds_valid) + len(ds_test),
    "filenames": {
        "train": [],
        "valid": [],
        "test": [],
    },
}
attribute_names = ds_train.attribute_names

# Loop dataloaders
idx = 0
image_sample = []
for split, dataset in zip(["train", "valid", "test"], [ds_train, ds_valid, ds_test]):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )
    for x, label in tqdm(dataloader, desc=split, ascii=True, ncols=80):
        x = x[0]
        label = label[0]
        # Process image and label/caption
        x = (x * 255).round().to(torch.uint8)
        y = torch.zeros(
            metadata["label_dim"], dtype=torch.float32
        )  # force unconditional
        c = []
        for i in range(len(label)):
            if label[i] > 0:
                c.append(attribute_names[i])
        c = " ".join(c)
        # Sample
        if split == "train" and torch.rand(1).item() < args.max_sample / len(ds_train):
            image_sample.append(x)
        # Save
        idx_str = str(idx).zfill(strlen_idx)
        fn = os.path.join(args.images_folder, idx_str[-3:], idx_str + ".pt")
        fullfn = os.path.join(args.dest_dir, fn)
        os.makedirs(os.path.split(fullfn)[0], exist_ok=True)
        torch.save(
            [idx, x, y, c, label], fullfn
        )  # Format should be always [i,x,y,c,extra0,extra1,...]
        metadata["filenames"][split].append(fn)
        # Increase counter
        idx += 1

# Fill in missing metadata info
image_sample = torch.stack(image_sample, dim=0)
image_sample = image_sample.float() / 255
metadata["image_size"] = image_sample.size()[1:]
metadata["image_mean"] = image_sample.mean((0, 2, 3), keepdim=True).squeeze(0)
metadata["image_std"] = image_sample.std((0, 2, 3), keepdim=True).squeeze(0)

# Save metadata
fullfn = os.path.join(args.dest_dir, "metadata.pt")
torch.save(metadata, fullfn)

# Save examples
image_sample = image_sample[torch.randperm(len(image_sample))[: args.num_examples]]
fullfn = os.path.join(args.dest_dir, "image_sample.png")
torchvision.utils.save_image(
    image_sample, fullfn, nrow=math.ceil(args.num_examples**0.5)
)

# Chmod
print("chmod...")
os.system("chmod -R 2777 " + args.dest_dir)

print("Done!")
