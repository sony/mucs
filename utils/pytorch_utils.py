import sys
import warnings
import torch
from lightning.fabric.utilities import AttributeDict
from lightning.fabric.loggers import TensorBoardLogger
import copy

###################################################################################################


def get_optimizer(conf, model):
    if conf.name.lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.name.lower().startswith("adam_"):
        _, b1, b2 = conf.name.split("_")
        b1, b2 = float(b1), float(b2)
        optim = torch.optim.Adam(model.parameters(), lr=conf.lr, betas=(b1, b2))
    elif conf.name.lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    elif conf.name.lower().startswith("adamw_"):
        _, b1, b2 = conf.name.split("_")
        b1, b2 = float(b1), float(b2)
        optim = torch.optim.AdamW(
            model.parameters(), lr=conf.lr, betas=(b1, b2), weight_decay=conf.wd
        )
    elif conf.name.lower() == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    else:
        raise NotImplementedError
    return optim


def get_scheduler(
    conf,
    optim,
    epochs=None,
    mode="min",
    warm_factor=0.005,
    plateau_factor=0.2,
):
    name = conf.sched.lower() if conf.sched is not None else "flat"
    sched_on_epoch = True
    if name == "flat" or name == "ctt":
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lambda epoch: 1.0,
        )
    elif name.startswith("plateau"):
        _, patience = name.split("_")
        patience = max(0, int(patience) - 1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode=mode,
            factor=plateau_factor,
            patience=patience,
        )
    elif name.startswith("poly"):
        _, power = name.split("_")
        power = float(power)
        sched = torch.optim.lr_scheduler.PolynomialLR(
            optim, total_iters=epochs, power=power
        )
    elif name.startswith("warmpoly"):
        _, nwarm, power = name.split("_")
        nwarm = max(1, min(int(nwarm), epochs - 1))
        power = float(power)
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.PolynomialLR(
            optim, total_iters=epochs - nwarm, power=power
        )
        sched = SequentialLR(optim, [s1, s2], [nwarm])
    elif name.startswith("sd"):
        _, ndec = name.split("_")
        ndec = max(1, min(int(ndec), epochs - 1))
        s1 = torch.optim.lr_scheduler.ConstantLR(
            optim, factor=1.0, total_iters=epochs - ndec
        )
        s2 = torch.optim.lr_scheduler.PolynomialLR(optim, power=2, total_iters=ndec)
        sched = SequentialLR(optim, [s1, s2], [epochs - ndec])
    elif name.startswith("wsd"):
        _, nwarm, ndec = name.split("_")
        nwarm = max(1, min(int(nwarm), epochs - 1))
        ndec = max(1, min(int(ndec), epochs - 1))
        assert epochs > nwarm + ndec
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.ConstantLR(
            optim, factor=1.0, total_iters=epochs - nwarm - ndec
        )
        s3 = torch.optim.lr_scheduler.PolynomialLR(optim, power=2, total_iters=ndec)
        sched = SequentialLR(optim, [s1, s2, s3], [nwarm, epochs - ndec])
    elif name.startswith("warmlin"):
        _, nwarm = name.split("_")
        nwarm = max(1, min(int(nwarm), epochs - 1))
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0, end_factor=1e-4, total_iters=epochs - nwarm - 1
        )
        sched = SequentialLR(optim, [s1, s2], [nwarm])
    elif name.startswith("warmctt"):
        _, nwarm = name.split("_")
        nwarm = max(1, min(int(nwarm), epochs - 1))
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.ConstantLR(
            optim, factor=1.0, total_iters=epochs - nwarm
        )
        sched = SequentialLR(optim, [s1, s2], [nwarm])
    else:
        raise NotImplementedError
    return sched, sched_on_epoch


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def step(self, cost=None):
        if cost is None:
            with warnings.catch_warnings():
                # otherwise it warns about passing the epoch number (?)
                warnings.simplefilter("ignore")
                super().step()
        else:
            super().step(cost)


###################################################################################################


def compute_weights_average(
    model,
    form="l1",
    considered_layers=(
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
    ),
):
    assert form in ("l1", "l2")
    num = torch.zeros(1, device=model.device)
    den = torch.zeros(1, device=model.device)
    for m in model.modules():
        if isinstance(m, considered_layers):
            w = m.weight
            n = m.weight.numel()
            if form == "l1":
                w = w.abs()
            elif form == "l2":
                w = w.pow(2)
            num += w.sum()
            den += n
    wd = num / den
    return wd


###################################################################################################


def get_logger(path):
    return TensorBoardLogger(
        root_dir=path,
        name="",
        version="",
        default_hp_metric=False,
    )


###################################################################################################


def create_state(
    model=None,
    ema=None,
    optim=None,
    sched=None,
    conf=None,
    epoch=None,
    cost=None,
):
    return AttributeDict(
        model=model,
        ema=ema,
        optim=optim,
        sched=sched,
        conf=conf,
        epoch=epoch,
        cost=cost,
    )


def retrieve_state(
    state,
    keys=["model", "ema", "optim", "sched", "conf", "epoch", "cost"],
):
    if type(keys) == str:
        keys = [keys]
        single = True
    else:
        single = False
    ret = []
    for key in keys:
        ret.append(state[key])
    if single:
        ret = ret[0]
    return ret


###################################################################################################


@torch.inference_mode()
def ema_update(net, ema, decay=0.9999):
    for p_ema, p_net in zip(ema.parameters(), net.parameters()):
        p_ema.mul_(decay).add_(p_net, alpha=1 - decay)
    for b_ema, b_net in zip(ema.buffers(), net.buffers()):
        b_ema.copy_(b_net)
    """
    Notes from https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    """


###################################################################################################


def copy_model(net, requires_grad=False):
    new_net = copy.deepcopy(net)
    new_net.requires_grad_(requires_grad)
    return new_net


###################################################################################################


class StackedRandomGenerator:

    def __init__(self, num=None, seeds=None, device="cpu"):
        super().__init__()
        assert (num is None and seeds is not None) or (
            num is not None and seeds is None
        ), "Either num or seeds must be provided (just one of them)"
        if seeds is None:
            seeds = [None] * num
        if type(seeds) != list:
            assert seeds.ndim == 1
        self.generators = []
        for seed in seeds:
            if seed is None:
                seed = torch.randint(0, 1 << 32, (1,)).item()
            else:
                seed = int(seed) % (1 << 32)
            self.generators.append(torch.Generator(device).manual_seed(seed))
        self.device = device

    def __len__(self):
        return len(self.generators)

    def rand(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.rand(size[1:], generator=gen, device=self.device, **kwargs)
                for gen in self.generators
            ]
        )

    def rand_like(self, input):
        return self.rand(input.shape, dtype=input.dtype, layout=input.layout)

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randn(size[1:], generator=gen, device=self.device, **kwargs)
                for gen in self.generators
            ]
        )

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(
                    *args, size=size[1:], generator=gen, device=self.device, **kwargs
                )
                for gen in self.generators
            ]
        )

    def randperm(self, length, **kwargs):
        return torch.stack(
            [
                torch.randperm(length, generator=gen, device=self.device, **kwargs)
                for gen in self.generators
            ]
        )

    def multinomial(self, weights, num_samples, **kwargs):
        return torch.stack(
            [
                torch.multinomial(weights, num_samples, generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


###################################################################################################


class LogDict:

    def __init__(self, d=None):
        self.reset()
        if d is not None:
            self.append(d)

    def reset(self):
        self.d = {}

    def get(self, keys=None, prefix="", suffix=""):
        if keys is None:
            keys = list(self.d.keys())
        elif type(keys) != list:
            return self.d[keys]
        d = {}
        for key in keys:
            new_key = prefix + key + suffix
            d[new_key] = self.d[key]
        return d

    def append(self, newd):
        assert type(newd) == dict
        for key, value in newd.items():
            value = value.cpu()
            if value.ndim == 0:
                value = torch.tensor([value], dtype=torch.float32)
            if key not in self.d:
                self.d[key] = value
            else:
                self.d[key] = torch.cat([self.d[key], value], dim=0)

    def sync_and_mean(self, fabric):
        fabric.barrier()
        for key in self.d.keys():
            self.d[key] = fabric.all_gather(self.d[key]).mean().item()


###################################################################################################


def gather_and_cat(fabric, x, dim=0):
    x = fabric.all_gather(x)
    x = torch.cat(torch.unbind(x, dim=0), dim=dim)
    return x


###################################################################################################


@torch.inference_mode()
def count_model_parameters(model):
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num


###################################################################################################


class MultiTensorList:

    def __init__(self):
        self.reset()

    def reset(self):
        self.l = []

    @torch.inference_mode()
    def append(self, l):
        if type(l) != list:
            l = [l]
        if len(self.l) == 0:
            for item in l:
                self.l.append([item])
        else:
            assert len(self.l) == len(l)
            for i in range(len(self.l)):
                self.l[i].append(l[i])

    @torch.inference_mode()
    def stack(self, dim=0):
        for i in range(len(self.l)):
            self.l[i] = torch.stack(self.l[i], dim=dim)

    @torch.inference_mode()
    def cat(self, dim=0):
        for i in range(len(self.l)):
            self.l[i] = torch.cat(self.l[i], dim=dim)

    @torch.inference_mode()
    def items(self, indices=None):
        if indices is None:
            return self.l
        ret = []
        for i in indices:
            ret.append(self.l[i])
        return ret


###################################################################################################
