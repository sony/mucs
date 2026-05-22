import sys
import torch
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from lib import lib_dinov2, lib_clip, lib_lpips, lib_sscd


class ImageMetrics:

    def __init__(self, which=[], batch_size=256, device="cpu", cache_dir=None):
        if type(which) != list:
            which = [which]
        for w in which:
            assert w in (
                "euc",
                "mse",
                "mae",
                "medae",
                "cos",
                "correl",
                "lpips",
                "lpipss",
                "dino",
                "clip",
                "comp",
                "bin",
                "ssim",
                "sscd",
            )
        self.which = which
        self.batch_size = batch_size
        self.device = device
        if "lpips" in which or "lpipss" in which or "comp" in which:
            self.lpips_loss = lib_lpips.get_model(device=self.device)
            self.lpips_loss.eval()
        if "clip" in which or "comp" in which:
            self.clip_model, self.clip_preproc = lib_clip.get_model(
                model_device=self.device, cache_dir=cache_dir
            )
            self.clip_model.eval()
        if "dino" in which or "comp" in which:
            self.dino_model = lib_dinov2.get_model(device=self.device)
            self.dino_model.eval()
        if "sscd" in which or "comp" in which:
            self.sscd_model = lib_sscd.get_model(
                device=self.device, cache_dir=cache_dir
            )
            self.sscd_model.eval()
        self.ssim = SSIM(
            data_range=1.0,
            kernel_size=7,
            sigma=1.5,
            gaussian_kernel=True,
            reduction="none",
        )

    def check_inputs(self, x, y):
        assert x.ndim == 4 and y.ndim == 4, "Format is b,c,h,w"
        assert (
            x.min() >= 0 and x.max() <= 1 and y.min() >= 0 and y.max() <= 1
        ), "Only accepts images between 0 and 1"
        for i in range(x.ndim):
            assert x.size(i) == y.size(i), f"Size mismatch in dim {i}"

    def calc(self, name, x, y):
        self.check_inputs(x, y)
        met = []
        for i in range(0, len(x), self.batch_size):
            lo, hi = i, min(i + self.batch_size, len(x))
            met.append(self.calc_(name, x[lo:hi], y[lo:hi]))
        met = torch.cat(met, dim=0)
        return met

    def calc_(self, name, x, y):
        self.check_inputs(x, y)
        x = x.to(self.device)
        y = y.to(self.device)

        if name == "euc":
            metric = (x - y).pow(2).sum((1, 2, 3)).sqrt()

        elif name == "mse":
            metric = (x - y).pow(2).mean((1, 2, 3))

        elif name == "mae":
            metric = (x - y).abs().mean((1, 2, 3))

        elif name == "medae":
            metric = (
                rearrange((x - y).abs(), "b c h w -> b (c h w)").median(dim=-1).values
            )

        elif name == "cos":
            x = torch.nn.functional.normalize(2 * x - 1, dim=(1, 2, 3))
            y = torch.nn.functional.normalize(2 * y - 1, dim=(1, 2, 3))
            metric = (x * y).sum((1, 2, 3))

        elif name == "correl":
            x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (
                x.std(dim=(1, 2, 3), keepdim=True) + 1e-4
            )
            y = (y - y.mean(dim=(1, 2, 3), keepdim=True)) / (
                y.std(dim=(1, 2, 3), keepdim=True) + 1e-4
            )
            metric = (x * y).mean((1, 2, 3))

        elif name == "lpips":
            metric = lib_lpips.get_dissim(self.lpips_loss, x, y)

        elif name == "lpipss":
            metric = lib_lpips.get_dissim(self.lpips_loss, x, y)
            metric = (1 - metric).clamp(min=-1)

        elif name == "clip":
            fx = lib_clip.get_features(self.clip_model, self.clip_preproc, x)
            fy = lib_clip.get_features(self.clip_model, self.clip_preproc, y)
            fx = torch.nn.functional.normalize(fx, dim=-1)
            fy = torch.nn.functional.normalize(fy, dim=-1)
            metric = (fx * fy).sum(-1)

        elif name == "dino":
            fx = lib_dinov2.get_features(self.dino_model, x)
            fy = lib_dinov2.get_features(self.dino_model, y)
            fx = torch.nn.functional.normalize(fx, dim=-1)
            fy = torch.nn.functional.normalize(fy, dim=-1)
            metric = (fx * fy).sum(-1)

        elif name == "sscd":
            fx = lib_sscd.get_features(self.sscd_model, x)
            fy = lib_sscd.get_features(self.sscd_model, y)
            fx = torch.nn.functional.normalize(fx, dim=-1)
            fy = torch.nn.functional.normalize(fy, dim=-1)
            metric = (fx * fy).sum(-1)

        elif name == "comp":
            metric = 0
            weight = 0
            for w, n in [
                (35, "ssim"),
                (5, "cos"),
                (26, "clip"),
                (26, "dino"),
                (8, "lpips"),
                # (1, "medae"),
                # (2, "mae"),
                # (2, "correl"),
            ]:
                if n == "medae":
                    tmp = (1 - self.calc_("medae", x, y) / 0.4).clamp(min=0)
                elif n == "mae":
                    tmp = (1 - self.calc_("mae", x, y) / 0.5).clamp(min=0)
                elif n == "ssim":
                    tmp = self.calc_("ssim", x, y).clamp(min=0)
                elif n == "cos":
                    tmp = self.calc_("cos", x, y).clamp(min=0)
                elif n == "correl":
                    tmp = self.calc_("correl", x, y).clamp(min=0)
                elif n == "lpips":
                    tmp = (1 - 1.33 * self.calc_("lpips", x, y)).clamp(min=0)
                elif n == "clip":
                    tmp = self.calc_("clip", x, y).clamp(min=0)
                elif n == "dino":
                    tmp = self.calc_("dino", x, y).clamp(min=0)
                metric += w * tmp
                weight += w
            metric /= weight

        elif name == "bin":
            metric = (self.calc("ssim", x, y) < 0.8).float()

        elif name == "ssim":
            metric = self.ssim(x, y)
            if len(x) == 1:
                metric = metric.unsqueeze(0)

        else:
            raise NotImplementedError
        return metric
