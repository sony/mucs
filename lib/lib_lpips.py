import sys, warnings
import torch
from torchvision import transforms
import lpips
from einops import rearrange

resize = transforms.Resize((224, 224))


def get_model(device="cpu"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        criterion = lpips.LPIPS(net="vgg")
    criterion.eval()
    return criterion.to(device)


def get_dissim(criterion, x, y):
    assert x.max() <= 1 and x.min() >= 0 and y.max() <= 1 and y.min() >= 0
    x = resize(2 * x - 1).clamp(min=-1, max=1)
    y = resize(2 * y - 1).clamp(min=-1, max=1)
    vals = criterion(x, y)
    return rearrange(vals, "n 1 1 1 -> n")
