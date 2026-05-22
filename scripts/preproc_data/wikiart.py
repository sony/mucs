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
parser.add_argument("--num_test", type=int, default=6000, required=False)
args, args_extra = parser.parse_known_args()

args.dest_dir = os.path.abspath(args.dest_dir)

# Create folder structure
if os.path.exists(args.dest_dir):
    print("Clean destination folder...")
    os.system("rm -rf " + args.dest_dir)
os.makedirs(args.dest_dir)

# Get datasets
print("Extracting data...")
meta, idmap, counts = torch.load(
    os.path.join(args.source_dir, "meta.pt"), weights_only=True, map_location="cpu"
)
strlen_idx = len(str(len(meta.keys()) - 1))
label_dim = 0
ndim = {}
for key in counts.keys():
    ndim[key] = len(counts[key])
    label_dim += ndim[key]

# Get validation and test splits
basenames = list(meta.keys())
random.seed(args.seed)
random.shuffle(basenames)
bn_train = basenames[args.num_test :]
bn_test = basenames[: args.num_test]

# Create metadata
metadata = {
    "dataset_name": "wikiart",
    "image_size": None,  # Computed after loading the data
    "image_mean": None,  # Computed after loading the data
    "image_std": None,  # Computed after loading the data
    "label_dim": label_dim,
    "num_samples": len(basenames),
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
for split, basenames in zip(["train", "test"], [bn_train, bn_test]):
    for bn in tqdm(basenames, desc=split, ascii=True, ncols=80):
        fields = bn.split("_")
        artist = fields[0]
        fullfn = os.path.join(args.source_dir, "images-proc", artist, bn + ".png")
        if not os.path.exists(fullfn):
            continue
        # Meta
        artist = artist.replace("-", " ")
        title = " ".join(fields[1:])
        title = title.replace("-", " ").replace("(", " (")
        # Load image
        x = torchvision.io.read_image(fullfn)  # Torch tensor between 0..256, uint
        x = resize(x)
        # Process labels/caption
        y = []
        c = []
        offset = 0
        for tag in ["artist", "genre", "style"]:
            if tag in meta[bn]:
                txt = meta[bn][tag]
                aux = torch.zeros(metadata["label_dim"], dtype=torch.float32)
                aux[offset + idmap[tag][txt]] = 1
                c.append(txt)
                y.append(aux)
            offset += ndim[tag]
        # Augment label: pairwise union + all_of_them item on top
        new_y = y[:]
        new_c = c[:]
        cc = " ".join(c)
        for i in range(0, len(y)):
            for j in range(i + 1, len(y)):
                new_y.append(y[i] + y[j])
                new_c.append(" ".join([c[i], c[j]]))
        y = torch.stack(new_y, dim=0)
        y = torch.cat([y.amax(0, keepdim=True), y], dim=0)
        c = [cc] + new_c
        # Sample
        if split == "train" and torch.rand(1).item() < args.max_sample / len(bn_train):
            image_sample.append(crop(x))
        # Save
        idx_str = str(idx).zfill(strlen_idx)
        fn = os.path.join(args.images_folder, idx_str[-3:], idx_str + ".pt")
        fullfn = os.path.join(args.dest_dir, fn)
        os.makedirs(os.path.split(fullfn)[0], exist_ok=True)
        torch.save(
            [idx, x, y, c, artist, title], fullfn
        )  # Format should be always [i,x,y,c,extra0,extra1,...]
        metadata["filenames"][split].append(fn)
        # Increase counter
        idx += 1
    # Print num found
    fns = metadata["filenames"][split]
    print(f"{len(basenames)} --> {len(fns)}")

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
