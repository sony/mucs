import sys, os
import torch
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

FIGNAME = "image_changes"
PATH = "pointer_to/logs/"
DATASETS = [
    ("cifar10_dit_edm_s1_k2.0_na20_sb0--pre", [0, 1, 2, 5, 6, 7, 8, 9, 10, 12]),
    ("artbench10_dit_edm_s1_k2.0_na20_sb0--pre", [0, 1, 5, 6, 7, 9, 10, 13, 14, 16]),
    ("coco_dit_edm_s1_k2.0_na20_sb0--pre", [0, 1, 3, 4, 7, 10, 11, 13, 15, 16]),
]
APPROACHES = [
    ("random", "Random"),
    ("clip", "CLIP"),
    ("dino", "DINO"),
    ("dtrak", "DTRAK"),
    ("das", "DAS"),
    ("finf", "Forward-INF"),
    ("usi", "AbU"),
    ("ssis", "MUCC"),
]
IMSIZE = 1.5


def run(cfg):
    print(FIGNAME)
    for dataset, select in DATASETS:
        sel = torch.LongTensor(select)

        dsname = dataset.split("_")[0]
        print("   Plot " + dsname + "...")
        info_img = []
        info_pos = []
        info_names = []
        # Load ref
        fn = os.path.join(PATH, dataset, "samples.pt")
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        i = 0
        x = (xgen[sel] * 0.5 + 0.5).clamp(min=0, max=1)
        for j in range(len(x)):
            info_img.append(rearrange(x[j], "c h w -> h w c").numpy())
            info_pos.append((i, j))
            info_names.append("Reference")
        # Load approaches
        for name, namestr in APPROACHES:
            folder = dataset.replace("--pre", "--post_" + name)
            fn = os.path.join(PATH, folder, "samples.pt")
            idx, x, y, xgen, extra = torch.load(
                fn, weights_only=True, map_location="cpu"
            )
            i += 1
            x = (xgen[sel] * 0.5 + 0.5).clamp(min=0, max=1)
            for j in range(len(x)):
                info_img.append(rearrange(x[j], "c h w -> h w c").numpy())
                info_pos.append((i, j))
                info_names.append(namestr)
        # Plot
        fig = plt.figure(
            figsize=(
                IMSIZE * len(sel) * cfg["cm2i"],
                IMSIZE * (len(APPROACHES) + 1) * cfg["cm2i"],
            ),
            dpi=200,
        )
        grid = ImageGrid(
            fig, 111, nrows_ncols=(len(APPROACHES) + 1, len(sel)), axes_pad=0
        )
        for ax, im, id, ns in zip(grid, info_img, info_pos, info_names):
            ax.imshow(im, interpolation="nearest")
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            i, j = id
            if j == 0:
                ax.set_ylabel(ns, fontsize=cfg["labelsize"])
        # Save
        fn = os.path.join(
            cfg["path_out"], "fig_" + FIGNAME + "_" + dsname + "." + cfg["ext_out"]
        )
        plt.savefig(fn, bbox_inches="tight")
