import os
import random
import torch, torchvision
import torchvision.transforms as transforms
import os, argparse
from tqdm import tqdm
import math
import open_clip
import torch.nn.functional as F
from data_utils import Cub2011
from torch.utils.data import ConcatDataset, random_split

"""
Datasets are stored in a folder with a metadata file (see below) and a
subfolder named "pt", nested by the last two indices...
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tmp_dir", type=str, default=None, required=True)
parser.add_argument("--dest_dir", type=str, default=None, required=True)
parser.add_argument("--images_folder", type=str, default="pt_files", required=False)
parser.add_argument("--max_sample", type=int, default=5000, required=False)
parser.add_argument("--num_examples", type=int, default=100, required=False)
parser.add_argument("--njobs", type=int, default=-1)
parser.add_argument("--image_size", type=int, default=128, required=False)   # 128 then interpolate to 512 for latent EDM
parser.add_argument("--path_clip_weights", type=str, default="pointer_to/cache/", required=False)
args, args_extra = parser.parse_known_args()

args.dest_dir = os.path.abspath(args.dest_dir)

# Create folder structure
if os.path.exists(args.dest_dir):
    print("Clean destination folder...")
    os.system("rm -rf " + args.dest_dir)
os.makedirs(args.dest_dir)

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k", cache_dir=args.path_clip_weights
)  # 'laion2b_s32b_b79k' for ViT-H-14
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer(
    "ViT-L-14", cache_dir=args.path_clip_weights
)  # ViT-L-14 - 768 dim; ViT-H-14 - 1024 dim embeddings
clip_model = clip_model.to(device="cuda")
print("CLIP Model loaded")

CUB_transforms = transforms.Compose(
    [
        transforms.Resize(
            args.image_size
        ),  # Resize the shorter side to 128/512, preserves the aspect ratio of image
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ]
)
print("Extracting data...")

# CUB dataset and CUB captions (taken from https://github.com/taoxugit/AttnGAN) downloaded in /group2/ds/data/birds/
train_dataset = Cub2011('/group2/ds/data/birds/', train=True, transform=CUB_transforms, download=False)
test_dataset = Cub2011('/group2/ds/data/birds/', train=False, transform=CUB_transforms, download=False)

print(
    "Number of samples in train set: ", len(train_dataset), "; test set: ", len(test_dataset)
)  # [train set:  5994 ; test set:  5794] -> total: 11788 -> we transform the partition to [train set: 10000; test set: 1788]

train_dataset2, ds_test = random_split(test_dataset, [4006, 1788])

ds_train = ConcatDataset([train_dataset, train_dataset2]) 
print("dataset partitions rearranged")

strlen_idx = len(str(len(ds_train) + len(ds_test) - 1))
print(
    "After changing partitions - Number of samples in train set: ", len(ds_train), "; test set: ", len(ds_test)
) 

# Create metadata
metadata = {
    "dataset_name": "CUB",
    "image_size": None,  # Computed after loading the data
    "image_mean": None,  # Computed after loading the data
    "image_std": None,  # Computed after loading the data
    "label_dim": 768,
    "num_samples": len(ds_train) + len(ds_test),
    "filenames": {
        "train": [],
        # "valid": [],
        "test": [],
    },
}

# Loop dataloaders
idx = 0
image_sample = []
for split, dataset in zip(["train", "test"], [ds_train, ds_test]):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )
    with torch.no_grad():
        for x, label in tqdm(dataloader, desc=split, ascii=True, ncols=80):
            x = x[0]
            # Process image and label/caption
            x = (x * 255).round().to(torch.uint8)
            label = [t[0] for t in label]
            # print(label)
            c = label
            text = tokenizer(label)
            text_features = clip_model.encode_text(text.to(device="cuda"))
            # # text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = F.normalize(text_features, dim=-1)
            y = text_features.cpu()

            # Sample
            if split == "train" and torch.rand(1).item() < args.max_sample / len(
                ds_train
            ):
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
