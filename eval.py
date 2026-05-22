import sys, argparse, os
from omegaconf import OmegaConf
import torch, lightning
import matplotlib.pyplot as plt
from sklearn import metrics as ir_metrics
from scipy import stats
import math

from lib import flexdataset, image_metrics
from utils import image_utils

###############################################################################

# Definitions
CM2I = 0.3937
METHOD_NAME = {
    "ref": "Ref",
    "random": "Rand",
    "dtrak": "DTRAK",
    "das": "DAS",
    "usi": "AbU",
    "ssis": "MUCC",
    "finf": "Forward-INF",
    "clip": "CLIP",
    "dino": "DINO",
    "cond": "Condition",
}
METHOD_COLOR = {
    "random": "k",
    # "das": "tab:blue",
}
METRIC_NAME = {
    "mae": "MAE",
    "medae": "MedAE",
    "ssim": "SSIM",
    "cos": "Cosine",
    "euc": "Euclidean",
    "lpips": "LPIPS",
    "lpipss": "LPIPS",
    "dino": "DINO",
    "clip": "CLIP",
    "sscd": "SSCD",
    "correl": "Correlation",
    "comp": "COMP",
}
METRICS_THAT_ARE_DISTANCES = ("mae", "medae", "euc", "lpips")
NUM_MAX_VIZ_IMG = 40
NUM_REF_ROLLS = 10
# HITRATE_QUANTILE = 0.1

###############################################################################

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--logpath", type=str, default="pointer_to/logs/", required=False)
parser.add_argument(
    "--cachepath", type=str, default="pointer_to/cache/", required=False
)
parser.add_argument("--jobs", type=str, default=None, required=True)
parser.add_argument("--methods", type=str, default=None, required=True)
parser.add_argument(
    "--metrics", type=str, default="ssim,sscd,lpipss,clip,dino", required=False
)
parser.add_argument("--fn_out", type=str, default="erase.png", required=False)
parser.add_argument("--device", type=str, default="cuda", required=False)
parser.add_argument("--calc_ref", action="store_true")
parser.add_argument("--pval", action="store_true")
parser.add_argument("--show_controls", action="store_true")
args = parser.parse_args()
args.jobs = args.jobs.split(",")
args.methods = args.methods.split(",")
args.metrics = args.metrics.split(",")
fn_base, fn_ext = os.path.splitext(args.fn_out)
assert "random" in args.methods, 'The "random" method must be included'

# Load model config
conf = OmegaConf.load(
    os.path.join(args.logpath, args.jobs[0] + "--pre", "configuration.yaml")
)

# Load data
print("Load data...")
dataset = flexdataset.Dataset(
    conf.data,
    "orig-test",
    seed=conf.seed,
    augment=False,
    return_extra=False,
)
data = {}
for job in args.jobs:
    data[job] = {}
    for method in args.methods:
        data[job][method] = {}
        for pp in ["pre", "post"]:
            if pp == "pre":
                fn = os.path.join(args.logpath, job + f"--pre", "samples.pt")
            else:
                fn = os.path.join(args.logpath, job + f"--post_{method}", "samples.pt")
            idx, x, y, xgen, extra = torch.load(
                fn, weights_only=True, map_location="cpu"
            )
            data[job][method][pp] = {
                "idx": idx,
                "x": x,
                "y": y,
                "xgen": xgen,
                "extra": extra,
            }
            print(f"\r  {job}  {method}  {pp}  ::  {len(x)}          ", end="")
    print()

###############################################################################


def get_num_attrib(jobname):
    num_attrib = None
    for s in jobname.split("_"):
        if s.startswith("na"):
            try:
                num_attrib = int(s[2:])
            except:
                pass
        if num_attrib is not None:
            break
    if num_attrib is None:
        sys.exit('ERROR: Couldn\'t find/parse "na" in jobname')
    return num_attrib


# Concatenate jobs and keep only useful data
print("Select data, concatenate, and denormalize...")
idx = {}
xgen = {}
for method in args.methods:
    idx[method] = {}
    xgen[method] = {}
    for grp in ["targets", "control"]:
        idx[method][grp] = {}
        xgen[method][grp] = {}
        for pp in ["pre", "post"]:
            # Concat jobs
            idx[method][grp][pp] = None
            xgen[method][grp][pp] = None
            for job in args.jobs:
                num_attrib = get_num_attrib(job)
                if grp == "control":
                    aux_idx = data[job][method][pp]["idx"][num_attrib:]
                    aux_xgen = data[job][method][pp]["xgen"][num_attrib:]
                else:
                    aux_idx = data[job][method][pp]["idx"][:num_attrib]
                    aux_xgen = data[job][method][pp]["xgen"][:num_attrib]
                if idx[method][grp][pp] == None:
                    idx[method][grp][pp] = aux_idx
                    xgen[method][grp][pp] = aux_xgen
                else:
                    idx[method][grp][pp] = torch.cat(
                        [idx[method][grp][pp], aux_idx], dim=0
                    )
                    xgen[method][grp][pp] = torch.cat(
                        [xgen[method][grp][pp], aux_xgen], dim=0
                    )
            # Denormalize
            xgen[method][grp][pp] = dataset.denormalize(xgen[method][grp][pp])

# Check idx
print("Check idx match...")
for method in args.methods:
    for grp in ["targets", "control"]:
        assert (
            idx[method][grp]["pre"] - idx[method][grp]["post"]
        ).abs().sum() < 0.5, f"\n{method} {grp} idxs do not match"

###############################################################################

# Ref seed distances
if args.calc_ref:
    print("Obtain reference images...")
    xgen["ref"] = {}
    for grp in ["targets", "control"]:
        aux = xgen["random"][grp]["pre"]
        tmp1 = []
        tmp2 = []
        for k in range(min(NUM_REF_ROLLS, len(aux) - 1)):
            tmp1.append(aux)
            tmp2.append(aux.roll(1 + k, dims=0))
        xgen["ref"][grp] = {}
        xgen["ref"][grp]["pre"] = torch.cat(tmp1, dim=0)
        xgen["ref"][grp]["post"] = torch.cat(tmp2, dim=0)
    args.methods = ["ref"] + args.methods

# Metrics
print("Init metrics...")
torch.set_float32_matmul_precision("medium")
fabric = lightning.Fabric(
    accelerator=args.device,
    devices=1,
    num_nodes=1,
    strategy=lightning.fabric.strategies.DDPStrategy(broadcast_buffers=False),
    precision="32",
)
fabric.launch()
fabric.barrier()
img_metrics = image_metrics.ImageMetrics(
    which=args.metrics, device=fabric.device, cache_dir=args.cachepath
)

print("Compute metrics...")
with torch.inference_mode():
    metrics = {}
    metnames = []
    for what in args.metrics:
        metnames.append(what)
        metrics[what] = {}
        for method in args.methods:
            metrics[what][method] = {}
            for grp in ["targets", "control"]:
                xpre = xgen[method][grp]["pre"]
                xpost = xgen[method][grp]["post"]
                metrics[what][method][grp] = img_metrics.calc(what, xpre, xpost).cpu()

###############################################################################

# Viz xgen
print("Viz images...")
nmax = min(NUM_MAX_VIZ_IMG, len(xgen[args.methods[0]]["targets"]["post"]))
for grp in ["targets", "control"]:
    x = []
    for method in args.methods:
        if method == "ref":
            continue
        namemethod = METHOD_NAME[method] if method in METHOD_NAME else method
        for pp in ["pre", "post"]:
            if pp == "pre" and method != "random":
                continue
            xg = xgen[method][grp][pp]
            xg = xg[:nmax]
            txt = " " if pp == "pre" and method == "random" else namemethod
            xpp = image_utils.text_as_image(txt, xg.size()[1:]).unsqueeze(0)
            x.append(xpp)
            x.append(xg)
    x = torch.cat(x, dim=0)
    fn = fn_base + "_images-" + grp + ".png"
    image_utils.save_image_grid(x, fn, ncols=nmax + 1)

###############################################################################

print("Evaluation...")
print("=" * 80)

# Load attrib ids
attrib_idx = {}
for method in args.methods:
    attrib_idx[method] = None
    for job in args.jobs:
        if method == "ref":
            fn = os.path.join(args.logpath, job + f"--post_random", "attrib-ids.pt")
        else:
            fn = os.path.join(args.logpath, job + f"--post_{method}", "attrib-ids.pt")
        aux = torch.load(fn, weights_only=True, map_location="cpu")
        if attrib_idx[method] is None:
            attrib_idx[method] = aux
        else:
            attrib_idx[method] = torch.cat([attrib_idx[method], aux], dim=0)

# Common IDs
print("   Common IDs:")
for i in range(len(args.methods)):
    method = args.methods[i]
    if method == "ref":
        continue
    namemethod = METHOD_NAME[method] if method in METHOD_NAME else method
    a = attrib_idx[method]
    overlap = []
    for j in range(0, len(a)):
        aj = set(a[j].tolist())
        for k in range(j + 1, len(a)):
            ak = set(a[k].tolist())
            overlap.append(len(aj & ak) / min(len(aj), len(ak)))
    overlap = torch.FloatTensor(overlap) * 100
    mu = overlap.mean()
    ci = 1.98 * overlap.std() / (len(overlap) ** 0.5)
    print(f"      {namemethod}\r\t\t\t= {mu:5.1f}% +- {ci:3.1f}")

# ID overlap across methods
overlap = torch.ones(len(args.methods), len(args.methods))
for i in range(0, len(args.methods)):
    method = args.methods[i]
    if method == "ref":
        continue
    a = attrib_idx[method]
    for j in range(i + 1, len(args.methods)):
        b = attrib_idx[args.methods[j]]
        ovr = torch.zeros(len(a))
        for k in range(len(a)):
            ak = set(a[k].tolist())
            bk = set(b[k].tolist())
            ovr[k] = len(ak & bk) / min(len(ak), len(bk))
        overlap[i, j] = ovr.mean()
        overlap[j, i] = overlap[i, j]

print("   Attributed IDs overlap:")
for i in range(overlap.size(0)):
    method = args.methods[i]
    if method == "ref":
        continue
    namemethod = METHOD_NAME[method] if method in METHOD_NAME else method
    print(f"      {namemethod}\r\t\t\t=", end="")
    for j in range(overlap.size(1)):
        if args.methods[j] == "ref":
            continue
        if i == j:
            print(f"     -  ", end="")
        else:
            print(f"  {100*overlap[i,j]:5.1f}%", end="")
    print()

###############################################################################

# Plot distributions
fig, axes = plt.subplots(
    len(args.metrics) // 2,
    2,
    figsize=(6 * CM2I * 2, 5 * CM2I * len(args.metrics) / 2),
    dpi=200,
)
i, j = 0, 0
for _, grp in enumerate(["targets"]):
    for _, metric in enumerate(args.metrics):
        metricname = METRIC_NAME[metric] if metric in METRIC_NAME else metric
        for method in args.methods:
            methodname = METHOD_NAME[method] if method in METHOD_NAME else method
            methodcolor = METHOD_COLOR[method] if method in METHOD_COLOR else None
            tmp = metrics[metric][method][grp].cpu()
            density = stats.gaussian_kde(tmp.numpy(), bw_method=0.25)
            if method == "euc" or method == "lpips":
                lo, hi = max(0, tmp.min() * 0.8), tmp.max() * 1.2
            else:
                lo, hi = 0, 1
            x = torch.linspace(lo, hi, 100).numpy()
            y = density(x)
            y /= sum(y)
            if method == "random":
                axes[i, j].fill_between(x, x * 0, y, color="tab:gray", alpha=0.27)
            axes[i, j].plot(x, y, label=methodname, color=methodcolor, linewidth=1)
        axes[i, j].legend(fontsize=5)
        axes[i, j].xaxis.set_tick_params(labelsize=5)
        axes[i, j].yaxis.set_tick_params(labelsize=5)
        axes[i, j].yaxis.set_ticks([])
        axes[i, j].set_xlabel(metricname, fontsize=6)
        axes[i, j].set_ylabel("Density", fontsize=6)
        j += 1
        if j == 2:
            i += 1
            if i >= len(args.metrics) // 2:
                break
            j = 0
plt.tight_layout()
plt.savefig(fn_base + "_distrib" + fn_ext)

###############################################################################

# Compute AUC
auc = {}
pval_t = {}  # https://doi.org/10.1256/003590002320603584
pval_c = {}
hr = {}
ratio, ratio_ci = {}, {}
for what in args.metrics:
    auc[what] = {}
    pval_t[what] = {}
    pval_c[what] = {}
    hr[what] = {}
    ratio[what], ratio_ci[what] = {}, {}
    for method in args.methods:
        if method == "ref":
            continue
        auc[what][method] = {}
        hr[what][method] = {}
        ratio[what][method], ratio_ci[what][method] = {}, {}
        for grp in ["targets", "control"]:
            # -----------------------------------------------------------------
            ref = metrics[what]["random"][grp].cpu()
            tmp = metrics[what][method][grp].cpu()
            if what in METRICS_THAT_ARE_DISTANCES:
                ref, tmp = -ref, -tmp
            # AUC
            ytrue = torch.cat([torch.zeros_like(tmp), torch.ones_like(ref)], dim=0)
            ypred = torch.cat([tmp, ref], dim=0)
            val = ir_metrics.roc_auc_score(ytrue.numpy(), ypred.numpy())
            auc[what][method][grp] = val
            # ratio
            r = 100 * (tmp / ref.median().clamp(min=0.01) - 1)
            # val = r.median().item()
            val = r.mean().item()
            ratio[what][method][grp] = val
            # mad = (r - r.median()).abs().median()
            # val = (
            #     stats.norm.ppf(1 - 0.05 / 2)
            #     * math.sqrt(torch.pi / (2 * len(r)))
            #     * mad
            #     / 0.6745
            # ).item()
            val = 1.98 * r.std().item() / (len(r) ** 0.5)
            ratio_ci[what][method][grp] = val
            # -----------------------------------------------------------------
            # ref = metrics[what]["ref"][grp].cpu()
            # tmp = metrics[what][method][grp].cpu()
            # hr
            # thres = ref.max()  # torch.quantile(ref, 0.99)
            val = (tmp < 0.95 * ref).float().mean().item()
            hr[what][method][grp] = val
            # -----------------------------------------------------------------
        # p-value
        pval_t[what][method] = {}
        pval_c[what][method] = {}
        for method2 in args.methods:
            x1 = metrics[what][method]["targets"].cpu()
            x2 = metrics[what][method2]["targets"].cpu()
            if what in METRICS_THAT_ARE_DISTANCES:
                x1, x2 = -x1, -x2
            _, val = stats.mannwhitneyu(x1.numpy(), x2.numpy(), alternative="less")
            pval_t[what][method][method2] = val
            x1 = metrics[what][method]["control"].cpu()
            x2 = metrics[what][method2]["control"].cpu()
            _, val = stats.mannwhitneyu(x1.numpy(), x2.numpy(), alternative="two-sided")
            pval_c[what][method][method2] = val

# Print & plot
print("=" * 80)
numchars = 21 if args.show_controls else 12
numnums = 15  # not touch

todo = [
    (auc, "AUC"),
    # (hr, "HR"),
    (ratio, "Ratio"),
]
if args.pval:
    todo.append((ratio_ci, "Ratio-ci"))
for res, name in todo:
    print(name)
    first = True
    for method in args.methods:
        if method == "ref":
            continue
        methodname = METHOD_NAME[method] if method in METHOD_NAME else method
        if first:
            print(
                "   "
                + "-" * (numchars * (1 + len(args.metrics)) - (numchars - numnums))
            )
            print("   " + " " * numchars, end=" ")
            for metric in args.metrics + ["mu"]:
                metricname = METRIC_NAME[metric] if metric in METRIC_NAME else metric
                print(
                    f"{metricname}" + " " * max(0, numchars - len(metricname)), end=""
                )
            print()
            print(
                "   "
                + "-" * (numchars * (1 + len(args.metrics)) - (numchars - numnums))
            )
            first = False
        print(f"   {methodname}" + " " * max(0, numchars - len(methodname)), end="")
        mu_target = []
        mu_control = []
        for metric in args.metrics:
            val_target = res[metric][method]["targets"]
            val_control = res[metric][method]["control"]
            mu_target.append(val_target)
            mu_control.append(val_control)
            if name in ("Ratio", "Ratio-ci"):
                if args.show_controls:
                    tmp = f"{val_target:+5.1f}% / {val_control:+5.1f}%"
                else:
                    tmp = f"{val_target:+5.1f}%"
            else:
                if args.show_controls:
                    tmp = f"{val_target:6.3f} / {val_control:6.3f}"
                else:
                    tmp = f"{val_target:6.3f}"
            print(
                tmp + " " * max(0, numchars - len(tmp)),
                end="",
            )
        mu_target = sum(mu_target) / len(mu_target)
        mu_control = sum(mu_control) / len(mu_control)
        if name in ("Ratio", "Ratio-ci"):
            if args.show_controls:
                tmp = f"{mu_target:+5.1f}% / {mu_control:+5.1f}%"
            else:
                tmp = f"{mu_target:+5.1f}%"
        else:
            if args.show_controls:
                tmp = f"{mu_target:6.3f} / {mu_control:6.3f}"
            else:
                tmp = f"{mu_target:6.3f}"
        print(tmp + " " * max(0, numchars - len(tmp)), end="")
        print()
    print("   " + "-" * (numchars * (1 + len(args.metrics)) - (numchars - numnums)))

if args.pval:
    print("p-values:")
    for metric in auc.keys():
        metricname = METRIC_NAME[metric] if metric in METRIC_NAME else metric
        print(f"   {metricname}:")
        for i, m1 in enumerate(pval_t[metric].keys()):
            methodname = METHOD_NAME[m1] if m1 in METHOD_NAME else m1
            print(
                f"      {methodname}" + " " * max(0, numchars - 4 - len(methodname)),
                end="",
            )
            for j, m2 in enumerate(pval_t[metric][m1].keys()):
                todo = [pval_t, pval_c] if args.show_controls else [pval_t]
                for k, pval in enumerate(todo):
                    if k == 0:
                        txt = "&"
                    elif j < i:
                        txt = "/"
                    else:
                        txt = " "
                    if j >= i:
                        txt += "    -    "
                        txt += "            "
                    else:
                        p = pval[metric][m1][m2]
                        txt += f" ${p:.1e}$ "
                        # txt = (
                        #     txt.replace("e", "\cdot 10^{")
                        #     .replace("$ ", "}$ ")
                        #     .replace("-0", " -")
                        # )
                    print(txt, end="")
            print()
    # Summary pval
    # print("Summary p-values:")
    # summary = {}
    # for metric in auc.keys():
    #     for m1 in pval_t[metric].keys():
    #         if m1 not in summary:
    #             summary[m1] = {}
    #         for m2 in pval_t[metric][m1].keys():
    #             if m2 not in summary[m1]:
    #                 summary[m1][m2] = []
    #             summary[m1][m2].append(pval_t[metric][m1][m2])
    # for i, m1 in enumerate(summary.keys()):
    #     methodname = METHOD_NAME[m1] if m1 in METHOD_NAME else m1
    #     print(
    #         f"   {methodname}" + " " * max(0, numchars - 4 - len(methodname)),
    #         end="",
    #     )
    #     for j, m2 in enumerate(summary[m1].keys()):
    #         txt = ""
    #         if j >= i:
    #             txt += "    -    "
    #             txt += "            "
    #         else:
    #             p = sum(summary[m1][m2]) / len(summary[m1][m2])
    #             # p = max(summary[m1][m2])
    #             txt += f" ${p:.1e}$ "
    #             # txt = (
    #             #     txt.replace("e", "\cdot 10^{")
    #             #     .replace("$ ", "}$ ")
    #             #     .replace("-0", " -")
    #             # )
    #         print(txt, end="")
    #     print()

print("=" * 80)
