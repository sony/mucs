import sys, os, argparse, time
from omegaconf import OmegaConf
import torch, lightning
import importlib

from lib import flexdataset
from utils import pytorch_utils, print_utils

###############################################################################
# Arguments
###############################################################################

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default=None, required=True)
parser.add_argument("--checkpoint", type=str, default=None, required=True)
parser.add_argument("--fn_samples", type=str, default=None, required=True)
parser.add_argument("--fn_out", type=str, default=None, required=True)
parser.add_argument("--model", type=str, default="ema", required=False)
parser.add_argument("--num_attrib", type=int, default=100, required=False)
parser.add_argument("--top_k_attrib", type=int, default=100, required=False)
parser.add_argument("--batch_size", type=int, default=128, required=False)
parser.add_argument("--num_workers", type=int, default=8, required=False)
args, args_extra = parser.parse_known_args()
log_path, _ = os.path.split(args.checkpoint)
method_path, _ = os.path.split(args.fn_out)

# Load model config
conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))

# Merge args into conf
conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args_extra))

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

# Print config
fabric.print(print_utils.config_string(args))
fabric.barrier()


###############################################################################
# Data
###############################################################################

fabric.print("Load generated data...")
idx, x, y, xgen, extra = torch.load(
    args.fn_samples, weights_only=True, map_location=fabric.device
)
fabric.print(f"  {xgen.size()}")

fabric.print("Load training data...")
ds_train = flexdataset.Dataset(
    conf.data,
    conf.data.partition_train,
    seed=conf.seed,
    augment=False,
    return_extra=False,
    verbose=fabric.is_global_zero,
)
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=False,
    persistent_workers=False,
    pin_memory=True,
)
dl_train = fabric.setup_dataloaders(dl_train)

###############################################################################
# Model
###############################################################################

# Init model
fabric.print("Init model...")
model_module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = model_module.Model(conf.model, ds_train.meta)
model = fabric.setup(model)
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
# Call attribution method
###############################################################################

# Load
fabric.print(f"Load module attribution::{args.method} ...")
attrib_module = importlib.import_module("attribution." + args.method)
attrib = attrib_module.Attributor(
    conf, fabric, model, method_path, device=fabric.device
)

# Run
tstart = time.time()
fabric.print("Run attribution...")
x_attrib = xgen[: args.num_attrib]
y_attrib = y[: args.num_attrib]
e_attrib = {}
for key in extra.keys():
    e_attrib[key] = extra[key][: args.num_attrib]
top_attribute_indices, all_idx, all_scores = attrib.run(
    dl_train,
    x_attrib,
    y_attrib,
    e_attrib,
    args.top_k_attrib,
)
assert (
    top_attribute_indices.ndim == 2
    and top_attribute_indices.size(0) == len(x_attrib)
    and top_attribute_indices.size(1) == args.top_k_attrib
)
top_attribute_indices = top_attribute_indices.cpu().long()
all_idx = all_idx.cpu().long()
all_scores = all_scores.cpu().float()
totaltime = (time.time() - tstart) / args.num_attrib  # in seconds

###############################################################################
# Save ids and quit
###############################################################################

fabric.print("Save...")
fn = args.fn_out
torch.save(top_attribute_indices, args.fn_out)
fn = os.path.splitext(args.fn_out)[0] + "_all.pt"
torch.save([top_attribute_indices, all_idx, all_scores], fn)

fabric.print("Done!")

fabric.print(f"Time = {totaltime:.1f} s")
fn = os.path.splitext(args.fn_out)[0] + "_log.txt"
with open(fn, "w") as fh:
    fh.write("Total time (in seconds) = " + str(totaltime) + "\n")
