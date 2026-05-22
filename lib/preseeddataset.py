import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, seeds, samples, labels):
        self.seeds = seeds
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.seeds[idx], self.samples[idx], self.labels[idx]
