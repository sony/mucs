import sys, os
import torch
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

FIGNAME = "image_pairs"
PATH = "pointer_to/logs/"
FILEINFO = [
    (
        "CIFAR10",
        "prepost",
        "cifar10_dit_edm_s1_k2.0_na20_sb0--pre",
        "cifar10_dit_edm_s1_k2.0_na20_sb0--post_random",
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10],
    ),
    (
        "ArtBench10",
        "prepost",
        "artbench10_dit_edm_s1_k2.0_na20_sb0--pre",
        "artbench10_dit_edm_s1_k2.0_na20_sb0--post_random",
        [0, 2, 3, 4, 6, 7, 8, 9, 11, 12],
    ),
    (
        "COCO",
        "prepost",
        "coco_dit_edm_s1_k2.0_na20_sb0--pre",
        "coco_dit_edm_s1_k2.0_na20_sb0--post_random",
        [0, 5, 6, 7, 8, 9, 10, 11, 12, 14],
    ),
]
IMSIZE = 1.5


def run(cfg):
    print(FIGNAME)

    # Loop
    print("   Plot...")
    for n, (ds_name, what, conf1, conf2, sel) in enumerate(FILEINFO):
        print(f"      {ds_name}, {what}")
        sel = torch.LongTensor(sel)
        if what == "prepost":
            label = ["Pre", "Post"]
        elif what == "half":
            label = ["Half-A", "Half-B"]
        # Load
        fn = os.path.join(PATH, conf1, "samples.pt")
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        x_pre = xgen[sel]
        x_pre = (x_pre * 0.5 + 0.5).clamp(min=0, max=1)
        fn = os.path.join(PATH, conf2, "samples.pt")
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        x_post = xgen[sel]
        x_post = (x_post * 0.5 + 0.5).clamp(min=0, max=1)
        # Plot
        fig = plt.figure(
            figsize=(
                IMSIZE * len(sel) * cfg["cm2i"],
                2.12 * IMSIZE * cfg["cm2i"],
            ),
            dpi=200,
        )
        grid = ImageGrid(fig, 111, nrows_ncols=(2, len(sel)), axes_pad=0)
        tmp = []
        idx = []
        for i, x in enumerate([x_pre, x_post]):
            for j in range(len(x)):
                tmp.append(rearrange(x[j], "c h w -> h w c").numpy())
                idx.append((i, j))
        for ax, im, id in zip(grid, tmp, idx):
            ax.imshow(im, interpolation="nearest")
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            i, j = id
            if j == 0:
                ax.set_ylabel(label[i], fontsize=cfg["fontsize"])
            # if i == 0 and j == len(sel) - 1:
            #     ax.set_title(ds_name, loc="right", fontsize=cfg["fontsize"] - 1)
            if i == 0 and j == 0:
                # ax.set_title(
                #     # chr(ord("@") + n + 1),
                #     "(" + chr(ord("`") + n + 1) + ")",
                #     loc="left",
                #     fontsize=cfg["fontsize"],
                #     fontweight="bold",
                # )
                ax.set_title(ds_name, loc="left", fontsize=cfg["fontsize"])
        # Save
        fn = os.path.join(
            cfg["path_out"], "fig_" + FIGNAME + "_" + str(n + 1) + "." + cfg["ext_out"]
        )
        plt.savefig(fn, bbox_inches="tight")
