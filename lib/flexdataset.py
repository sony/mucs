import sys, os
import torch, torchvision
import math

PERFORM_ID_CHECK = False

###################################################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        conf,
        split,
        seed=44,
        verbose=False,
        znorm=False,
        augment=False,
        return_extra=False,
    ):
        # Params
        self.split = split
        self.rng = torch.Generator("cpu").manual_seed(seed)
        self.verbose = verbose
        self.znorm = znorm
        self.return_extra = return_extra
        self.augment = augment
        # Load metadata
        self.meta = torch.load(
            os.path.join(conf.path, "metadata.pt"), weights_only=True, mmap="cpu"
        )
        # Select partition
        if split.startswith("full"):
            self.filenames = []
            for part in ["train", "valid", "test"]:
                if part in self.meta["filenames"]:
                    self.filenames += self.meta["filenames"][part]
        elif split.startswith("orig-"):
            part = split.split("-")[-1]
            self.filenames = self.meta["filenames"][part]
        elif split.startswith("half-"):
            part = split.split("-")[-1]
            filenames = self.meta["filenames"]["train"]
            ids = torch.randperm(len(filenames), generator=self.rng)
            if part == "a":
                ids = ids[: len(ids) // 2]
            else:
                ids = ids[len(ids) // 2 :]
            self.filenames = [filenames[i] for i in ids]
        elif split.startswith("sub-"):
            _, part, sp = split.split("-")
            filenames = self.meta["filenames"][sp]
            ids = torch.randperm(len(filenames), generator=self.rng)
            if part == "a":
                ids = ids[: len(ids) // 2]
            elif part == "b":
                ids = ids[len(ids) // 2 :]
            filenames = [filenames[i] for i in ids]
            if self.verbose:
                print("[Loading labels; this may take time]")
            labels = []
            for fn in filenames:
                fn = os.path.join(conf.path, fn)
                y = torch.load(fn, weights_only=True, mmap="cpu")[2]
                labels.append(torch.argmax(y).item())
            max_items = {
                0: 1,
                1: 4,
                2: 10,
                3: 50,
                4: 150,
                5: 500,
                6: 2000,
            }
            num_items = {}
            for key in max_items.keys():
                num_items[key] = 0
            self.filenames = []
            for fn, cla in zip(filenames, labels):
                if cla not in num_items or num_items[cla] < max_items[cla]:
                    self.filenames.append(fn)
                    if cla in num_items:
                        num_items[cla] += 1
        else:
            raise NotImplementedError
        # Expand path
        for i in range(len(self.filenames)):
            self.filenames[i] = os.path.join(conf.path, self.filenames[i])
        # ID check
        if PERFORM_ID_CHECK:
            if self.verbose:
                print("  [Performing ID check; this may take some time]")
            for fullfn in self.filenames:
                idx = int(os.path.splitext(os.path.split(fullfn)[-1])[0])
                i = torch.load(fullfn, weights_only=True, mmap="cpu")[0]
                assert idx == i
        # Transforms
        self.normalize = torchvision.transforms.Normalize(
            mean=self.meta["image_mean"] if self.znorm else 0.5,
            std=self.meta["image_std"] if self.znorm else 0.5,
            inplace=True,
        )
        self.crop_size = min(self.meta["image_size"][-1], self.meta["image_size"][-2])
        self.crop_random = torchvision.transforms.RandomCrop(self.crop_size)
        self.crop_center = torchvision.transforms.CenterCrop(self.crop_size)
        # Print
        if self.verbose:
            print(f"  {conf.name} :: {self.split} :: {len(self.filenames)} images")

    def __len__(self):
        return len(self.filenames)

    def remove_selected_datapoints(self, fn_ids):
        tmp = torch.load(fn_ids, weights_only=True, mmap="cpu")
        id_set = set(tmp.long().flatten().tolist())
        new_filenames = []
        for fullfn in self.filenames:
            # We leverage the fact that the filename is directly the image idx
            idx = int(os.path.splitext(os.path.split(fullfn)[-1])[0])
            if idx not in id_set:
                new_filenames.append(fullfn)
        if self.verbose:
            print(f"  Removed items ({len(self.filenames)}-->{len(new_filenames)})")
        self.filenames = new_filenames[:]

    def set_num_datapoints(self, n, seed=None):
        if seed is not None:
            rng = torch.Generator().manual_seed(seed)
        else:
            rng = self.rng
        ids = []
        while len(ids) < n:
            ids += torch.randperm(len(self.filenames), generator=rng).tolist()
        ids = ids[:n]
        tmp = self.filenames[:]
        self.filenames = [tmp[i] for i in ids]
        if self.verbose:
            print(f"  Set dataset size to {n} (sampled data)")

    def __getitem__(self, idx):
        # Load
        tmp = torch.load(self.filenames[idx], weights_only=True, mmap="cpu")
        i, x, y, c = tmp[:4]
        # Proc x
        x = self.normalize(x.float() / 255)
        if self.augment:
            x = self.crop_random(x)
        else:
            x = self.crop_center(x)
        # Proc y
        if y.ndim > 1:
            if self.augment:
                r = torch.randint(len(y), (1,), generator=self.rng).item()
                y = y[r]
                c = c[r]
            else:
                y = y[0]
                c = c[0]
        # Return
        if self.return_extra:
            if len(tmp) > 4:
                return [i, x, y, c] + tmp[4:]
            return i, x, y, c
        return i, x, y

    def denormalize(self, x, to_uint=False):
        assert x.ndim >= 3
        if self.znorm:
            mean = self.meta["image_mean"].to(dtype=x.dtype, device=x.device)
            std = self.meta["image_std"].to(dtype=x.dtype, device=x.device)
            while mean.ndim < x.ndim:
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
        else:
            mean = 0.5
            std = 0.5
        x = (x * std + mean).clamp(min=0, max=1)
        if to_uint:
            x = (255.5 * x).to(torch.uint8)
        return x


###################################################################################################
