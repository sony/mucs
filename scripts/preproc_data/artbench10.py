import sys, os, argparse
import torch, torchvision
from tqdm import tqdm
import math
import random

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
parser.add_argument("--seed", type=int, default=45, required=False)
args, args_extra = parser.parse_known_args()

args.dest_dir = os.path.abspath(args.dest_dir)

# Create folder structure
if os.path.exists(args.dest_dir):
    print("Clean destination folder...")
    os.system("rm -rf " + args.dest_dir)
os.makedirs(args.dest_dir)

# Get datasets
print("Extracting data...")
with open(os.path.join(args.source_dir, "metadata.csv")) as f:
    lines = f.readlines()[1:]
names, labels, splits = [], [], []
for line in lines:
    fields = line.split(",")
    names.append(fields[0])
    labels.append(fields[6])
    splits.append(fields[7])
strlen_idx = len(str(len(names) - 1))

# Create metadata
metadata = {
    "dataset_name": "artbench",
    "image_size": None,  # Computed after loading the data
    "image_mean": None,  # Computed after loading the data
    "image_std": None,  # Computed after loading the data
    "label_dim": 10,
    "num_samples": len(names),
    "filenames": {
        "train": [],
        # "valid": [],
        "test": [],
    },
}

# Transforms
resize = torchvision.transforms.Resize(args.image_size)
crop = torchvision.transforms.CenterCrop(args.image_size)

# Loop dataloaders
idx = 0
image_sample = []
labelmap = {}
for name, label, split in tqdm(
    zip(names, labels, splits), desc="Proc", ascii=True, ncols=80
):
    fullfn = os.path.join(args.source_dir, split, label, name)
    if not os.path.exists(fullfn):
        continue
    # Load image
    x = torchvision.io.read_image(fullfn)  # Torch tensor between 0..256, uint
    x = resize(x)
    # Process labels/caption
    if label not in labelmap:
        labelmap[label] = len(labelmap)
    y = torch.zeros(metadata["label_dim"], dtype=torch.float32)
    y[labelmap[label]] = 1
    c = label
    # Sample
    if split == "train" and torch.rand(1).item() < args.max_sample / len(names):
        image_sample.append(crop(x))
    # Save
    idx_str = str(idx).zfill(strlen_idx)
    fn = os.path.join(args.images_folder, idx_str[-3:], idx_str + ".pt")
    fullfn = os.path.join(args.dest_dir, fn)
    os.makedirs(os.path.split(fullfn)[0], exist_ok=True)
    torch.save(
        [idx, x, y, c], fullfn
    )  # Format should be always [i,x,y,c,extra0,extra1,...]
    metadata["filenames"][split].append(fn)
    # Increase counter
    idx += 1
print(
    str(len(metadata["filenames"]["train"]))
    + " train, "
    + str(len(metadata["filenames"]["test"]))
    + " test"
)

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
