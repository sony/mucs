import sys, os, argparse
from omegaconf import OmegaConf
import torch, lightning, torchinfo
import importlib

from lib import flexdataset, augment, trainer_module
from utils import pytorch_utils, print_utils

###############################################################################
# Arguments
###############################################################################

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=str, default=None, required=True)
parser.add_argument("--use_scratch", action="store_true")
parser.add_argument("--half_epochs", action="store_true")
args, args_extra = parser.parse_known_args()

# Load config
conf = OmegaConf.load(args.conf)

# Merge extra arguments into conf
conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args_extra))
conf.path.logs = conf.slurm.logpath
conf.data.path = conf.path.data
if args.half_epochs:
    conf.training.num_epochs = int(0.5 * conf.training.num_epochs)

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
    loggers=pytorch_utils.get_logger(conf.path.logs),
)
fabric.launch()

# Common seed to have same model everywhere (for different rands per GPU see re-seed below)
fabric.print("Seed...")
fabric.barrier()
fabric.seed_everything(conf.seed, workers=True)

# Use scratch folder?
if args.use_scratch:
    fabric.print("Copying data to scratch folder...")
    lastpath = os.path.basename(os.path.normpath(conf.path.data))
    if fabric.is_global_zero:
        os.system("cp -r " + conf.path.data + " " + os.environ["SLURM_SCRATCH"])
    fabric.barrier()
    conf.path.data = os.path.join(os.environ["SLURM_SCRATCH"], lastpath)
    conf.data.path = conf.path.data

# Print config
fabric.print(print_utils.config_string(conf))

###############################################################################
# Data
###############################################################################

# Datasets
fabric.print("Load data...")
ds_train = flexdataset.Dataset(
    conf.data,
    conf.data.partition_train,
    seed=conf.seed,
    augment=True,
    verbose=fabric.is_global_zero,
)
if conf.data.subsample_train is not None:
    ds_train.set_num_datapoints(conf.data.subsample_train, seed=conf.seed)
if conf.data.remove_selected_datapoints is not None:
    ds_train.remove_selected_datapoints(conf.data.remove_selected_datapoints)
ds_valid = flexdataset.Dataset(
    conf.data,
    conf.data.partition_valid,
    seed=conf.seed,
    augment=False,
    verbose=fabric.is_global_zero,
)

# Dataloaders
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=conf.training.batch_size,
    shuffle=True,
    num_workers=conf.data.num_workers,
    drop_last=True,
    persistent_workers=False,
    pin_memory=True,
)
dl_valid = torch.utils.data.DataLoader(
    ds_valid,
    batch_size=conf.training.batch_size,
    shuffle=False,
    num_workers=conf.data.num_workers,
    drop_last=True,
    persistent_workers=False,
    pin_memory=True,
)
dl_train, dl_valid = fabric.setup_dataloaders(dl_train, dl_valid)

"""
# Data viz
import torchvision

for batch in dl_train:
    idx, x, y = batch
    x = x[: 8 * 8]
    xx = ds_train.denormalize(x)
    torchvision.utils.save_image(xx, "sample_batch.png", nrow=8)
    sys.exit()
# """

# Augmentations
augmentations = augment.Augmentations(conf.augmentation)

###############################################################################
# Model + Optim + Sched
###############################################################################

# Save config
fabric.print("Save config...")
if fabric.is_global_zero:
    fn = os.path.join(conf.path.logs, "configuration.yaml")
    print("  " + fn)
    with open(fn, "w") as fh:
        fh.write(OmegaConf.to_yaml(conf))

# Init model
fabric.print("Init model...")
model_module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = model_module.Model(conf.model, ds_train.meta)
if fabric.is_global_zero:
    torchinfo.summary(
        model,
        depth=1,
        col_names=["num_params", "params_percent"],
        col_width=20,
        mode="train",
    )
model = fabric.setup(model)
if conf.training.ema is not None and conf.training.ema > 0:
    fabric.print("[Using EMA]")
    ema = pytorch_utils.copy_model(model)
else:
    ema = None

# Init optimizer & scheduler
fabric.print("Init optimizer...")
optim = pytorch_utils.get_optimizer(conf.training.optim, model)
optim = fabric.setup_optimizers(optim)
sched, _ = pytorch_utils.get_scheduler(
    conf.training.optim,
    optim,
    epochs=conf.training.num_epochs,
    mode=conf.training.monitor.mode,
)

###############################################################################
# Checkpointing
###############################################################################

# Is there a previous checkpoint?
fn_ckpt = None
if conf.checkpoint is not None:
    fn_ckpt = conf.checkpoint
else:
    fn_ckpt_last, _, _ = trainer_module.get_checkpoint_names(conf.path.logs)
    if os.path.exists(fn_ckpt_last):
        fabric.print("Found last checkpoint")
        fn_ckpt = fn_ckpt_last

# Restore
if fn_ckpt is None:
    start_epoch = 0
    cost_best = None
else:
    fabric.print("Loading checkpoint...")
    state = pytorch_utils.create_state(
        model=model, ema=ema, optim=optim, sched=sched, conf=conf
    )
    fabric.load(fn_ckpt, state)
    model, ema, optim, sched, conf, start_epoch, cost_best = (
        pytorch_utils.retrieve_state(state)
    )
    fabric.print(f"  Loaded {fn_ckpt}")
    fabric.print(f"  [epoch={start_epoch}, cost_best={cost_best:.3f}]")

###############################################################################
# Train loop
###############################################################################

# Re-seed with global_rank to have truly different augmentations per GPU
fabric.print("Re-seed...")
fabric.barrier()
fabric.seed_everything(
    (start_epoch + 1) * (conf.seed + fabric.global_rank), workers=True
)

# Trainer init
fabric.print("Init trainer...")
trainer = trainer_module.Trainer(
    conf,
    fabric,
    model,
    ema,
    optim,
    sched,
    augment=augmentations,
    limit_batches=conf.limit_batches,
)

# Trainer launch
fabric.print("Train...")
stop, _ = trainer.run(
    dl_train,
    dl_valid,
    start_epoch=start_epoch,
    cost_best=cost_best,
)

# Done
if stop is None:
    fabric.print("Done!")
else:
    fabric.print(stop + " Stop.")

###############################################################################
