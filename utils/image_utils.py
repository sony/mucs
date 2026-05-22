import sys
import torch, torchvision
import math
from matplotlib import pyplot as plt
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
import numpy as np

###############################################################################


def save_image_grid(images, filename, ncols=None):
    assert images.ndim == 4
    images = images.cpu()
    if images.dtype == torch.uint8:
        images = images.float() / 255
    if ncols is None:
        ncols = math.ceil(len(images) ** 0.5)
    torchvision.utils.save_image(images, filename, nrow=ncols)


###############################################################################


def save_image_tworows(images1, images2, xlabel, ylabel, filename):
    assert (
        images1.ndim == 4
        and images1.ndim == images2.ndim
        and len(images1) == len(images2)
        and images1.dtype == images2.dtype
    )
    images1, images2 = images1.cpu(), images2.cpu()
    if images1.dtype == torch.uint8:
        images1 = images1.float() / 255
        images2 = images2.float() / 255
    images1 = rearrange(images1, "b c h w -> b h w c")
    images2 = rearrange(images2, "b c h w -> b h w c")
    n = len(images1)
    # plot
    fig, ax = plt.subplots(
        nrows=2, ncols=n, sharex=True, sharey=True, figsize=(0.85 * n, 1.5)
    )
    for i, img in enumerate([images1, images2]):
        for j in range(n):
            ax[i, j].imshow(img[j], origin="upper", interpolation="nearest")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 1:
                ax[i, j].set_xlabel(xlabel + " " + str(j), fontsize=8)
            if j == 0:
                ax[i, 0].set_ylabel(ylabel + " " + str(i + 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)


###############################################################################


def save_image_nrows(image_list, label_list, filename):
    fig, ax = plt.subplots(
        nrows=len(image_list),
        ncols=len(image_list[0]),
        sharex=True,
        sharey=True,
        figsize=(0.85 * len(image_list[0]), 1.5 * len(image_list)),
    )
    for i, (images, label) in enumerate(zip(image_list, label_list)):
        images = images.cpu()
        if images.dtype == torch.uint8:
            images = images.float() / 255
        images = rearrange(images, "b c h w -> b h w c")
        for j in range(len(images)):
            ax[i, j].imshow(images[j], origin="upper", interpolation="nearest")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel(label, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)


###############################################################################


def save_image_caption_list(captions, images, filename):
    assert len(captions) == len(images)
    assert images.ndim == 4 or images.ndim == 5
    images = images.cpu()
    if images.ndim == 4:
        images = rearrange(images, "b c h w -> b h w c")
    else:
        images = rearrange(images, "b1 b2 c h w -> b1 b2 h w c")
    # get dims
    nrows = len(images)
    if images.ndim == 4:
        ncols = 1
    else:
        ncols = images.size(1)
    # plot
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharey=True,
        figsize=(max(6, 1.5 * ncols), 1.05 * nrows),
    )
    for i in range(nrows):
        if ncols == 1:
            ax[i].imshow(images[i], origin="upper", interpolation="nearest")
            ax[i].axis("off")
            ax[i].set_title(captions[i], fontsize=9)
        else:
            for j in range(ncols):
                ax[i, j].imshow(images[i, j], origin="upper", interpolation="nearest")
                ax[i, j].axis("off")
            ax[i, ncols // 2].set_title(captions[i], fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)


###############################################################################


def text_as_image(
    text, imgsize, fontsize=7, foreground_color="black", background_color="white"
):
    c, h, w = imgsize
    if c == 1:
        img = Image.new("RGB", (w, h), background_color)
    elif c == 3:
        img = Image.new("RGB", (w, h), background_color)
    else:
        raise NotImplementedError
    draw = ImageDraw.Draw(img)
    draw.text((w * 0.05, h * 0.4), text, fill=foreground_color, font_size=fontsize)
    x = np.array(img)
    x = torch.FloatTensor(x) / 255.0
    x = rearrange(x, "h w c -> c h w")
    return x
