import sys, os
import torch
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.getcwd())
from lib import image_metrics

FIGNAME = "example_distribs"
PATH = "pointer_to/logs/"
CONFIG_FILE = "cifar10_dit_edm_s???_k2.0_na20_sb0"
SEEDS = [1, 2, 3, 4]
METHOD = ["ssis", "finf"]
METHODNAMES = ["MUCS", "Forward-INF"]
METRIC = "ssim"
METRICNAME = "Similarity (SSIM)"
KERNEL_BW = 0.25


def run(cfg):
    print(FIGNAME)

    # Load images
    print("   Load...")
    x_rand_pre = []
    x_rand_post = []
    x_met1_post = []
    x_met2_post = []
    for seed in SEEDS:
        fn = os.path.join(PATH, CONFIG_FILE + "--pre", "samples.pt")
        fn = fn.replace("???", str(seed))
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        x_rand_pre.append(xgen)
        fn = os.path.join(PATH, CONFIG_FILE + "--post_random", "samples.pt")
        fn = fn.replace("???", str(seed))
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        x_rand_post.append(xgen)
        fn = os.path.join(PATH, CONFIG_FILE + "--post_" + METHOD[0], "samples.pt")
        fn = fn.replace("???", str(seed))
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        x_met1_post.append(xgen)
        fn = os.path.join(PATH, CONFIG_FILE + "--post_" + METHOD[1], "samples.pt")
        fn = fn.replace("???", str(seed))
        idx, x, y, xgen, extra = torch.load(fn, weights_only=True, map_location="cpu")
        x_met2_post.append(xgen)
    x_rand_pre = torch.stack(x_rand_pre, dim=0)
    x_rand_post = torch.stack(x_rand_post, dim=0)
    x_met1_post = torch.stack(x_met1_post, dim=0)
    x_met2_post = torch.stack(x_met2_post, dim=0)
    x_rand_pre = (x_rand_pre * 0.5 + 0.5).clamp(min=0, max=1)
    x_rand_post = (x_rand_post * 0.5 + 0.5).clamp(min=0, max=1)
    x_met1_post = (x_met1_post * 0.5 + 0.5).clamp(min=0, max=1)
    x_met2_post = (x_met2_post * 0.5 + 0.5).clamp(min=0, max=1)

    # Targets and controls
    x_rand_pre_target, _ = torch.chunk(x_rand_pre, 2, dim=1)
    x_rand_post_target, _ = torch.chunk(x_rand_post, 2, dim=1)
    x_met1_post_target, _ = torch.chunk(x_met1_post, 2, dim=1)
    x_met2_post_target, _ = torch.chunk(x_met2_post, 2, dim=1)
    x_rand_pre_target = rearrange(x_rand_pre_target, "n b c h w -> (b n) c h w")
    x_rand_post_target = rearrange(x_rand_post_target, "n b c h w -> (b n) c h w")
    x_met1_post_target = rearrange(x_met1_post_target, "n b c h w -> (b n) c h w")
    x_met2_post_target = rearrange(x_met2_post_target, "n b c h w -> (b n) c h w")

    # Metric
    print("   Calc metric...")
    metrics = image_metrics.ImageMetrics(which=[METRIC])
    dist_rand_target = metrics.calc(METRIC, x_rand_pre_target, x_rand_post_target)
    dist_met1_target = metrics.calc(METRIC, x_rand_pre_target, x_met1_post_target)
    dist_met2_target = metrics.calc(METRIC, x_rand_pre_target, x_met2_post_target)

    # Plot
    print("   Plot...")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(cfg["width"] * cfg["cm2i"] * 2, cfg["height"] * cfg["cm2i"] * 0.9),
        dpi=200,
    )
    for i, (d_rand, d_meth, method_name) in enumerate(
        [
            (dist_rand_target, dist_met1_target, METHODNAMES[0]),
            (dist_rand_target, dist_met2_target, METHODNAMES[1]),
        ]
    ):
        x = torch.linspace(0, 1, 100).numpy()
        xref = 0.81
        ymax = 0
        # Curves
        density = stats.gaussian_kde(d_rand.numpy(), bw_method=KERNEL_BW)
        y = density(x)
        y /= sum(y)
        ymax = max(ymax, max(y))
        axes[i].fill_between(x, x * 0, y, color="tab:gray", alpha=0.25)
        axes[i].plot(x, y, label="Random", color="k", linewidth=cfg["linewidth"])
        density = stats.gaussian_kde(d_meth.numpy(), bw_method=KERNEL_BW)
        y = density(x)
        y /= sum(y)
        ymax = max(ymax, max(y))
        axes[i].fill_between(x, x * 0, y, color="tab:blue", alpha=0.25)
        axes[i].plot(
            x, y, label=method_name, color="tab:blue", linewidth=cfg["linewidth"]
        )
        axes[i].vlines(
            xref,
            0,
            ymax * 1.02,
            color="tab:brown",
            linestyles="dashed",
            linewidth=cfg["linewidth"],
            alpha=0.5,
        )
        for sign in [-1, 1]:
            axes[i].arrow(
                xref + sign * 0.02,
                ymax * 0.75,
                sign * 0.02,
                0,
                color="tab:brown",
                linewidth=cfg["linewidth"] * 0.333,
                alpha=0.4,
            )
        # Boundaries
        axes[i].legend(fontsize=cfg["labelsize"])
        axes[i].xaxis.set_tick_params(labelsize=cfg["labelsize"])
        axes[i].yaxis.set_tick_params(labelsize=cfg["labelsize"])
        axes[i].yaxis.set_ticks([])
        # axes[i].set_xlim(0, 1)
        # axes[i].set_ylim(0, ymax * 1.05)
        axes[i].set_xlabel(METRICNAME, fontsize=cfg["fontsize"])
        axes[i].set_ylabel("Density", fontsize=cfg["labelsize"])
    # Images
    imsize = 0.12
    for i, o, x_met in [(0, 0, x_met1_post_target), (3, 0.42, x_met2_post_target)]:
        for x, y, img in [
            (o + 0.40, 0.70, x_rand_pre_target[i]),
            (o + 0.41, 0.62, x_rand_post_target[i]),
            (o + 0.28, 0.24, x_rand_pre_target[i]),
            (o + 0.31, 0.30, x_met[i]),
        ]:
            axim = fig.add_axes([x, y, imsize, imsize])
            axim.imshow(
                rearrange(img, "c h w -> h w c").numpy(), interpolation="nearest"
            )
            axim.axis("off")
            # axim.xaxis.set_ticks([])
            # axim.yaxis.set_ticks([])
    # Save
    fn = os.path.join(cfg["path_out"], "fig_" + FIGNAME + "." + cfg["ext_out"])
    plt.savefig(fn, bbox_inches="tight")
