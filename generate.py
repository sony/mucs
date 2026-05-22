import sys, os, argparse
import importlib
from omegaconf import OmegaConf
import torch, lightning
from einops import rearrange
import random

from lib import flexdataset, preseeddataset
from utils import print_utils, pytorch_utils, image_utils

NUM_MAX_VIZ_IMG = 40

###############################################################################
# Arguments
###############################################################################

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, required=True)
parser.add_argument("--fn_out", type=str, default=None, required=True)
parser.add_argument("--pre_seeds", type=str, default=None, required=False)
parser.add_argument("--model", type=str, default="ema", required=False)
parser.add_argument("--num_gen", type=int, default=1000, required=False)
parser.add_argument("--split", type=str, default="orig-test", required=False)
parser.add_argument("--batch_size", type=int, default=250, required=False)
parser.add_argument("--num_workers", type=int, default=8, required=False)
args, args_extra = parser.parse_known_args()
log_path, _ = os.path.split(args.checkpoint)
# method_path, _ = os.path.split(args.fn_out)


# Load model config
conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))

# Merge args into conf
args_other = []
for ae in args_extra:
    args_other.append(ae)
conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args_other))

###############################################################################
# PyTorch/Fabric
###############################################################################

# Init pytorch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)

# Init fabric
fabric = lightning.Fabric(
    accelerator="cuda",
    devices=conf.slurm.ngpus,
    num_nodes=conf.slurm.nnodes,
    strategy=lightning.fabric.strategies.DDPStrategy(broadcast_buffers=False),
    precision=conf.slurm.precision,
)
fabric.launch()
fabric.seed_everything(conf.seed, workers=True)  # common seed same model in all GPUs
random.seed(conf.seed)
fabric.barrier()

# Print config
fabric.print(print_utils.config_string(args, bar="=-"))

###############################################################################
# Data
###############################################################################

# Default dataset
fabric.print("Load data...")
ds_samples = flexdataset.Dataset(
    conf.data,
    args.split,
    seed=conf.seed,
    augment=False,
    return_extra=False,
    verbose=fabric.is_global_zero,
)

# Get data loader depending on pre_seeds
if args.pre_seeds is None:
    ds_samples.set_num_datapoints(args.num_gen, seed=conf.seed)
    loader = torch.utils.data.DataLoader(
        ds_samples,
        batch_size=args.batch_size,
        shuffle=False,  # already shuffled when doing set_num_datapoints
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=False,
        pin_memory=True,
    )
else:
    idx_pre, x_pre, y_pre, _, _ = torch.load(
        args.pre_seeds, weights_only=True, map_location="cpu"
    )
    loader = torch.utils.data.DataLoader(
        preseeddataset.Dataset(idx_pre, x_pre, y_pre),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=False,
        pin_memory=True,
    )
    fabric.print("  [Using the seeds selected before]")

# Setup
loader = fabric.setup_dataloaders(loader)

###############################################################################
# Model
###############################################################################

# Init model
fabric.print("Init model...")
model_module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = model_module.Model(conf.model, ds_samples.meta)
model = fabric.setup(model)
model.mark_forward_method("generate")
fabric.print(f"  [{pytorch_utils.count_model_parameters(model)*1e-6:.1f}M parameters]")
if conf.training.ema is not None and conf.training.ema > 0:
    ema = pytorch_utils.copy_model(model)
else:
    ema = None

# Load model
fabric.print("Load checkpoint...")
state = pytorch_utils.create_state(model=model, ema=ema)
fabric.load(args.checkpoint, state)
model, epoch, best = pytorch_utils.retrieve_state(state, [args.model, "epoch", "cost"])
fabric.print(f"  [epoch={epoch}, cost_best={best:.3f}]")
model.eval()

###############################################################################
# Generation
###############################################################################

# Inits
progbar = lambda it, desc=None: print_utils.myprogbar(
    it, desc=desc, leave=True, doit=fabric.is_global_zero
)

# Generate data
fabric.print("Generating...")
with torch.inference_mode():
    # Examples generation
    all_idx = []
    all_x = []
    all_xgen = []
    all_y = []
    all_extra = {
        "t_steps": [],
        "expect1": [],
        "noised1": [],
        "expect2": [],
        "noised2": [],
        "logits1": [],
        "logits2": [],
    }  # Include all the keywords we want to export
    #    they should be (at least) tensors of ndim=2, first dim being the batch one

    # Loop data
    fabric.barrier()
    for batch in progbar(loader, desc="Batches"):
        idx, x, y = batch
        # Generate
        xgen, extra = model.generate(
            y=y,
            rng=pytorch_utils.StackedRandomGenerator(seeds=idx, device=x.device),
            num_samples=1,
            return_extra=True,
        )
        # Append
        all_idx.append(idx.cpu())
        all_x.append(x.cpu())
        all_xgen.append(xgen.cpu())
        all_y.append(y.cpu())
        for key in all_extra.keys():
            if key not in extra:
                all_extra[key].append(torch.zeros(len(xgen), 1))
            else:
                all_extra[key].append(extra[key].cpu())

    all_idx = torch.cat(all_idx, dim=0)
    all_x = torch.cat(all_x, dim=0)
    all_xgen = torch.cat(all_xgen, dim=0)
    all_y = torch.cat(all_y, dim=0)
    for key in all_extra.keys():
        all_extra[key] = torch.cat(all_extra[key], dim=0)
    # Gather
    fabric.print("Gather...")
    fabric.barrier()
    all_idx = pytorch_utils.gather_and_cat(fabric, all_idx)
    all_x = pytorch_utils.gather_and_cat(fabric, all_x)
    all_xgen = pytorch_utils.gather_and_cat(fabric, all_xgen)
    all_y = pytorch_utils.gather_and_cat(fabric, all_y)
    for key in all_extra.keys():
        all_extra[key] = pytorch_utils.gather_and_cat(fabric, all_extra[key])
        # print(key, all_extra[key].shape)
        # t_steps torch.Size([600, 33])
        # expect1 torch.Size([600, 32, 3, 32, 32])
        # noised1 torch.Size([600, 32, 3, 32, 32])

    # Save
    if fabric.is_global_zero:
        # Filter too similar images
        if args.pre_seeds is None:
            # Prune
            if len(all_xgen) > args.num_gen:
                # Create similarity matrix
                tmp = rearrange(all_xgen, "b c h w -> b (c h w)")
                tmp = torch.nn.functional.normalize(tmp, dim=-1)
                sim = tmp @ tmp.T
                sim.fill_diagonal_(-1)  # ignore self-similarity
                # Prune
                fabric.print("Pruning...")
                remaining = [g for g in range(len(all_xgen))]
                while len(remaining) > args.num_gen:
                    # find maximum similarity value and its indices
                    remove_idx = (torch.argmax(sim) // sim.size(1)).item()
                    sim[remove_idx, :] = -1
                    sim[:, remove_idx] = -1
                    remaining.remove(remove_idx)
                random.shuffle(remaining)
                fabric.print(f"  {len(all_xgen)} --> {len(remaining)}")
                all_idx = all_idx[remaining]
                all_x = all_x[remaining]
                all_xgen = all_xgen[remaining]
                all_y = all_y[remaining]
                for key in all_extra.keys():
                    all_extra[key] = all_extra[key][remaining]
            # Compute some stats and save them
            tmp = rearrange(all_xgen, "b c h w -> b (c h w)")
            tmp = torch.nn.functional.normalize(tmp, dim=-1)
            sim = tmp @ tmp.T
            # cross_seed_sim = torch.unique(torch.tril(sim, diagonal=-1))
            cross_seed_sim = sim[torch.triu(torch.ones_like(sim) == 1, 1)]
            sim_max = torch.max(cross_seed_sim).item()
            sim_mean = torch.mean(cross_seed_sim).item()
            sim_std = torch.std(cross_seed_sim).item()
            fabric.print(
                f"  [Smax={sim_max:.2f}, Smean={sim_mean:.2f}, Sstd={sim_std:.2f}]"
            )
            fn = os.path.splitext(args.fn_out)[0] + "_cross_seed_sim.pt"
            torch.save([cross_seed_sim, sim_max, sim_mean, sim_std], fn)
        fabric.print("Save...")
        # pt save
        torch.save([all_idx, all_x, all_y, all_xgen, all_extra], args.fn_out)
        # png save
        fn = os.path.splitext(args.fn_out)[0] + "_grid.png"
        tmp = all_xgen
        if len(tmp) > NUM_MAX_VIZ_IMG:
            tmp = torch.cat(
                [tmp[: NUM_MAX_VIZ_IMG // 2], tmp[-NUM_MAX_VIZ_IMG // 2 :]], dim=0
            )
        tmp = ds_samples.denormalize(tmp)
        image_utils.save_image_grid(tmp, fn, ncols=NUM_MAX_VIZ_IMG // 4)


fabric.print("Done.")

###############################################################################
