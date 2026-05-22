import sys, os, argparse
import torch, torchvision
from tqdm import tqdm
import math

"""
Datasets are stored in a folder with a metadata file (see below) and a
subfolder named "pt", nested by the last two indices...
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tmp_dir", type=str, default=None, required=True)
parser.add_argument("--dest_dir", type=str, default=None, required=True)
parser.add_argument("--images_folder", type=str, default="pt_files", required=False)
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
download = not os.path.exists(os.path.join(args.tmp_dir, "cifar-10-batches-py"))
ds_train = torchvision.datasets.CIFAR10(
    root=args.tmp_dir,
    train=True,
    download=download,
    transform=torchvision.transforms.ToTensor(),
)
ds_test = torchvision.datasets.CIFAR10(
    root=args.tmp_dir,
    train=False,
    download=download,
    transform=torchvision.transforms.ToTensor(),
)
strlen_idx = len(str(len(ds_train) + len(ds_test) - 1))

# Create metadata
metadata = {
    "dataset_name": "cifar10",
    "image_size": None,  # Computed after loading the data
    "image_mean": None,  # Computed after loading the data
    "image_std": None,  # Computed after loading the data
    "label_dim": 10,
    "num_samples": len(ds_train) + len(ds_test),
    "filenames": {
        "train": [],
        # "valid": [],
        "test": [],
    },
}
classtext = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# Loop dataloaders
idx = 0
image_sample = []
for split, dataset in zip(["train", "test"], [ds_train, ds_test]):
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
        label = label[0].item()
        # Process image and label/caption
        x = (x * 255).round().to(torch.uint8)
        y = torch.zeros(metadata["label_dim"], dtype=torch.float32)
        y[label] = 1  # in general, we want to ensure L2-normalized y
        c = classtext[label]
        # Sample
        if split == "train" and torch.rand(1).item() < args.max_sample / len(ds_train):
            image_sample.append(x)
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
