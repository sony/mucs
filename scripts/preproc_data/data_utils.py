import sys, os
import torch
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

###############################################################################


class Cub2011(Dataset):
    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self, root, train=True, transform=None, loader=default_loader, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        caption_filepath = sample.filepath[:-3] + "txt"
        caption_path = os.path.join(self.root, "text/", caption_filepath)
        with open(caption_path, "r") as f:
            # captions = f.readlines()
            captions = [line.strip() for line in f]

        if self.transform is not None:
            img = self.transform(img)

        # return img, target
        return img, captions  # , target


###############################################################################


class CelebA(Dataset):

    def __init__(self, root, split="train", transform=None, loader=default_loader):
        self.path = os.path.join(root, "img_align_celeba")
        self.transform = transform
        self.loader = loader
        # Load partitions
        with open(os.path.join(root, "anno", "list_eval_partition.txt")) as fh:
            lines = fh.readlines()
        filenames = {}
        for line in lines:
            fn, sp = line[:-1].split(" ")
            sp = int(sp)
            if (
                (split == "train" and sp == 0)
                or (split == "valid" and sp == 1)
                or (split == "test" and sp == 2)
            ):
                filenames[fn] = None
        # Load attributes
        with open(os.path.join(root, "anno", "list_attr_celeba.txt")) as fh:
            lines = fh.readlines()
        lines = lines[1:]
        for i in range(len(lines)):
            lines[i] = lines[i][:-1]
            while "  " in lines[i]:
                lines[i] = lines[i].replace("  ", " ")
            while lines[i][0] == " ":
                lines[i] = lines[i][1:]
            while lines[i][-1] == " ":
                lines[i] = lines[i][:-1]
        self.attribute_names = lines[0].split(" ")
        lines = lines[1:]
        self.filenames = []
        self.attributes = []
        for line in lines:
            fields = line.split(" ")
            fn, attr = fields[0], fields[1:]
            if fn in filenames:
                y = torch.zeros(len(attr))
                for i in range(len(attr)):
                    y[i] = 1 if int(attr[i]) > 0 else 0
                self.filenames.append(fn)
                self.attributes.append(y)
        self.attributes = torch.stack(self.attributes, dim=0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.path, self.filenames[idx])
        img = self.loader(path)
        attr = self.attributes[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, attr


###############################################################################
