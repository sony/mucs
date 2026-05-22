import sys
import torch
import matplotlib.pyplot as plt


def draw_violin(
    axs,
    data,  # data=[vector1,vector2,...]
    positions,  # positions=[i1,i2,...]
    side,  # "lo", "high", or "both"
    color,
    orientation="vertical",
    width=0.7,
    alpha=0.5,
    showextrema=False,
    scatter=True,
):
    parts = axs.violinplot(
        data,
        positions=positions,
        orientation=orientation,
        side=side,
        showextrema=showextrema,
        showmeans=False,
        showmedians=False,
        widths=width,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(alpha)
    if scatter:
        if side == "high":
            sign = 1
        else:
            sign = -1
        for dat, pos in zip(data, positions):
            r = torch.randn_like(dat).abs()
            axs.scatter(
                pos + sign * (r * width * 0.04 + width * 0.01),
                dat,
                s=width * 0.4,
                color=color,
            )
    # axs.scatter(
    #     coord[:, 0] + sign * 0.01,
    #     data.max(1)[0],
    #     s=0.6,
    #     color=color,
    #     marker="+",
    # )
    # axs.scatter(
    #     coord[:, 0] + sign * 0.01,
    #     data.min(1)[0],
    #     s=0.6,
    #     color=color,
    #     marker="+",
    # )
