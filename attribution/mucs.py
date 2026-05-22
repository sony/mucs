import sys, os
import torch
from omegaconf import OmegaConf
import importlib
from einops import rearrange, repeat
from copy import deepcopy
from scipy import stats

from lib import augment, image_metrics
from lib import tensor_ops as tops
from utils import print_utils, pytorch_utils

DEBUG_MODE = False
DEBUG_PARAM_NAMES = False
DEBUG_NUM_INFER = 1200
DEBUG_NUM_SAMPLES = 10


###########################################################################
###########################################################################


class Attributor(object):

    def __init__(
        self,
        conf,
        fabric,
        model,
        method_path,
        device="cpu",
        seed=45,
        batch_size=None,
        num_noises=100,
        lr_factor=10,
        lamb=0.2,
        loss_null="auto",
        tolerance=0.05,
        max_steps=None,
        frozen_params=[
            "encoder.",
            "decoder.",
            "s_embedder.",
            # "y_embedder.",
            ".xdrop",
            "a_embedder.",
            "x_embedder.",
            "pos_enc.",
            ".attention.qkv",
            ".attention.proj",
            # ".mlp.",
            ".adapter.",
            "unpatchify.",
            "last.",
            ".bias",
        ],
        use_aug=False,
        use_same_noise_train=False,
        use_same_noise_infer=True,
        use_instance_weights=False,
        t_sched_train=None,
        t_sched_infer=0.7,
        use_loss_ratio=True,
        eps=1e-3,
    ):
        self.conf = conf
        self.fabric = fabric
        self.model = model.cpu()
        self.method_path = method_path
        self.device = device

        self.batch_size = num_noises if batch_size is None else batch_size
        self.num_noises = num_noises
        self.conf.training.optim.lr /= lr_factor
        self.lamb = lamb
        self.loss_null = loss_null
        self.tol = tolerance
        self.max_steps = max_steps
        self.frozen_params = frozen_params
        self.use_aug = use_aug
        self.use_same_noise_train = use_same_noise_train
        self.use_same_noise_infer = use_same_noise_infer
        self.use_instance_weights = use_instance_weights
        self.t_sched_train = t_sched_train
        self.t_sched_infer = t_sched_infer
        self.use_loss_ratio = use_loss_ratio
        self.eps = eps

        assert self.device == self.fabric.device
        if self.use_instance_weights:
            metric = "dino"
            self.image_metrics = image_metrics.ImageMetrics(
                which=metric, device=self.device
            )
            self.image_sim_calc = lambda x1, x2: self.image_metrics.calc(metric, x1, x2)
        if self.use_aug:
            self.augment = augment.Augmentations(self.conf.augmentation)
        self.progbar = lambda it, desc=None: print_utils.myprogbar(
            it, desc=desc, leave=True, doit=self.fabric.is_global_zero
        )
        self.fabric.seed_everything(seed, workers=True)

        if DEBUG_PARAM_NAMES:
            for name, pars in self.model.module.named_parameters():
                print(name)
            sys.exit()

    ###########################################################################

    def run(
        self,
        dl_train,
        xgen,
        ygen,
        extra,
        top_k_attrib,
    ):
        if DEBUG_MODE:
            print("\n!!!!! DEBUG MODE !!!!!\n")

        # Dataloaders
        dl_inference = torch.utils.data.DataLoader(
            dl_train.dataset,
            batch_size=max(1, (16 * self.batch_size) // self.num_noises),
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )
        dl_unlearn = torch.utils.data.DataLoader(
            dl_train.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        dl_inference, dl_unlearn = self.fabric.setup_dataloaders(
            dl_inference, dl_unlearn
        )

        # Auto-setup?
        if self.t_sched_train == "auto" or self.t_sched_infer == "auto":
            tmp = self.estimate_t_sched(dl_unlearn)
            if self.t_sched_train == "auto":
                self.t_sched_train = tmp
            if self.t_sched_infer == "auto":
                self.t_sched_infer = tmp
        if self.loss_null == "auto":
            self.loss_null = self.estimate_null_loss(dl_unlearn)

        # Inference (original model)
        print("Original losses...")
        model = deepcopy(self.model.module).to(self.device)
        all_idx, loss_t, noiseinfo = self.compute_inference_losses(model, dl_inference)
        loss_t = loss_t.unsqueeze(0)  # 1,Nt,Nn

        # Loop samples
        loss_u = []
        for i in range(len(xgen)):
            print(f"Sample {i+1}/{len(xgen)}...")
            # Unlearn
            model = deepcopy(self.model.module).to(self.device)
            model = self.compute_unlearning(model, dl_unlearn, xgen[i], ygen[i])
            # Inference (unlearned model)
            _, losses, _ = self.compute_inference_losses(
                model, dl_inference, noiseinfo=noiseinfo
            )
            loss_u.append(losses)
            if DEBUG_MODE and i + 1 >= DEBUG_NUM_SAMPLES:
                break
        loss_u = torch.stack(loss_u, dim=0)  # Ng,Nt,Nn
        if DEBUG_MODE:
            print("Loss sizes =", loss_t.size(), loss_u.size())
            print(loss_t[0, :20])
            print(loss_u[0, :20])

        # Scores
        scores = self.compute_scores(loss_t, loss_u)
        if DEBUG_MODE:
            print("Scores")
            print(scores[0, :20])
            basename, _ = os.path.splitext(os.path.basename(__file__))
            path = self.method_path.replace("_" + basename, "_combi")
            fn = os.path.join(path, "attrib-ids_all.pt")
            _, _, refscores = torch.load(fn, weights_only=True, map_location="cpu")
            refscores = refscores[: scores.size(0), : scores.size(1)]
            rho = []
            for i in range(scores.size(0)):
                rho.append(
                    stats.spearmanr(
                        scores[i].cpu().numpy(), refscores[i].cpu().numpy()
                    ).statistic
                )
            print("size =", list(scores.size()))
            print(f"rho = {sum(rho) / len(rho):.3f}")
            sys.exit()

        # Get top-k
        ret_idx = get_topk(top_k_attrib, all_idx, scores)

        return ret_idx, all_idx, scores

    ###########################################################################

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_inference_losses(self, model, dl, noiseinfo=None, force_y=None):
        # Need noiseinfo?
        if noiseinfo is None:
            for batch in dl:
                idx, x, y = batch[:3]
                break
            noiseinfo = self.get_noise_info([self.num_noises] + list(x.size()[1:]))
        # Prepare
        model.eval()
        noise, sigma = noiseinfo
        # Loop
        idxs = []
        losses = []
        num = 0
        for batch in self.progbar(dl, desc="Inference"):
            # Prepare
            idx, x, y = batch[:3]
            bsz = len(x)
            x = repeat(x, "b c h w -> (n b) c h w", n=len(noise))
            y = repeat(y, "b d -> (n b) d", n=len(noise))
            if force_y is not None:
                y = repeat(force_y, "d -> b d", b=len(y))
            if self.use_same_noise_infer:
                n = repeat(noise, "n c h w -> (n b) c h w", b=bsz)
                s = repeat(sigma, "n -> (n b)", b=bsz)
            else:
                n, s = self.get_noise_info(x.size())
            # Predict
            xhat = self.get_model_pred(model, x, n, s, y)
            # Loss
            loss = self.get_model_loss(x, xhat, s)
            loss = rearrange(loss, "(n b) -> b n", b=bsz)
            idxs.append(idx)
            losses.append(loss)
            num += len(loss)
            if DEBUG_MODE and num >= DEBUG_NUM_INFER:
                break
        idxs = torch.cat(idxs, dim=0)
        losses = torch.cat(losses, dim=0)
        return idxs, losses, noiseinfo

    def compute_unlearning(self, model, dl, xgen_i, ygen_i):
        # Prepare
        model.eval()  # (on purpose, for unlearning)
        optim = pytorch_utils.get_optimizer(self.conf.training.optim, model)
        optim = self.fabric.setup_optimizers(optim)
        for name, pars in model.named_parameters():
            pars.requires_grad = not self.is_frozen_param(name)
        # Loop
        epoch = 0
        step = 0
        while True:
            for batch in self.progbar(dl, desc=f"Unlearn {epoch+1:d}"):
                # Prepare
                xr = batch[1]
                yr = batch[2]
                ar = None
                xf = repeat(xgen_i, "c h w -> b c h w", b=len(xr))
                yf = repeat(ygen_i, "d -> b d", b=len(xr))
                af = None
                if self.use_aug:
                    xr, ar = self.get_augmentations(xr)
                    xf, af = self.get_augmentations(xf)
                if self.use_instance_weights:
                    wr = self.get_instance_weight(dl.dataset, xr, yr, xf, yf).clone()
                else:
                    wr = 1
                # Predict
                nr, sr = self.get_noise_info(xr.size(), training=True)
                if self.use_same_noise_train:
                    nf, sf = nr, sr
                else:
                    nf, sf = self.get_noise_info(xf.size(), training=True)
                xrhat = self.get_model_pred(model, xr, nr, sr, yr, a=ar)
                xfhat = self.get_model_pred(model, xf, nf, sf, yf, a=af)
                # Losses
                lr = self.get_model_loss(xr, xrhat, sr)
                lr = (wr * lr).mean()
                lf = self.get_model_loss(xf, xfhat, sf)
                lf = lf.clamp(max=self.loss_null).mean()
                loss = lr - self.lamb * lf
                # Done?
                print(f"    L = {lr:.3f} vs {lf:.3f} ", end="")
                if (
                    self.max_steps is None and lf >= (1 - self.tol) * self.loss_null
                ) or (self.max_steps is not None and step >= self.max_steps):
                    return model
                # Step
                optim.zero_grad(set_to_none=True)
                self.fabric.backward(loss)
                optim.step()
                step += 1
            epoch += 1

    @torch.inference_mode()
    def compute_scores(self, loss_t, loss_u):
        if not self.use_same_noise_infer:
            loss_u = loss_u.mean(-1)
            loss_t = loss_t.mean(-1)
        # Loss comparison
        if self.use_loss_ratio:
            scores = (loss_u - loss_t) / (loss_u + loss_t + self.eps)
        else:
            scores = loss_u - loss_t
        if self.use_same_noise_infer:
            scores = scores.mean(-1)
        return scores

    ###########################################################################

    @torch.inference_mode()
    def get_augmentations(self, x):
        x, a = self.augment(x)
        return x, a

    @torch.inference_mode()
    def get_instance_weight(self, ds, xi, yi, xg, yg, q=0.1):
        xi = ds.denormalize(xi)
        xg = ds.denormalize(xg)
        sim = self.image_sim_calc(xi, xg)
        if not self.model.unconditional:
            sim += tops.cosine_similarity(yi, yg).clamp(min=0).sqrt()
        thres = torch.quantile(sim, 1 - q)
        weight = (sim <= thres).float()
        weight /= weight.mean()
        return weight

    @torch.inference_mode()
    def get_noise_info(self, size, training=False):
        t_sched = self.t_sched_train if training else self.t_sched_infer
        if t_sched is None:
            s = torch.randn(size[0], device=self.device)
            sigma = (self.model.sigma_P_std * s + self.model.sigma_P_mean).exp()
        else:
            num = int(size[0] / t_sched)
            if training:
                s = torch.rand(num, device=self.device).sort(descending=False).values
            else:
                s = torch.linspace(0, 1, num, device=self.device)
            sigma_min = self.conf.model.sampler.sigma_min
            sigma_max = self.conf.model.sampler.sigma_max
            rho = self.conf.model.sampler.rho
            sigma = (
                sigma_max ** (1 / rho)
                + s * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
            sigma = sigma[: size[0]]
        noise = rearrange(sigma, "n -> n 1 1 1") * torch.randn(size, device=self.device)
        return noise, sigma

    def get_model_pred(self, model, x, n, s, y, a=None):
        return model(x + n, s, y=y, a=a)

    def get_model_loss(self, x, xhat, s, reduction="pixels"):
        w = (s**2 + self.model.sigma_data**2) / (s * self.model.sigma_data) ** 2
        w = rearrange(w, "b -> b 1 1 1")
        loss = w * (xhat - x).pow(2)
        if reduction is None or reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "pixels":
            loss = loss.mean((-1, -2, -3))
        else:
            raise NotImplementedError
        return loss

    def is_frozen_param(self, name):
        for txt in self.frozen_params:
            if txt in name:
                return True
        return False

    ###########################################################################

    @torch.inference_mode()
    def estimate_t_sched(self, dl, num_consec=3, qtl=0.05):
        distances = []
        prev_x = []
        for batch in self.progbar(dl, desc="Distances"):
            _, x, _ = batch[:3]
            if len(prev_x) > 0:
                a = rearrange(x, "b c h w -> b (c h w)")
                b = rearrange(torch.cat(prev_x, dim=0), "b c h w -> b (c h w)")
                dist = tops.pairwise_distance_matrix(a, b, mode="neuc")
                distances.append(dist.view(-1).cpu())
            prev_x.append(x)
            if len(prev_x) > num_consec:
                prev_x = prev_x[1:]
            if DEBUG_MODE and len(distances) > 10:
                break
        distances = torch.cat(distances, dim=0)
        tmp1, tmp2 = self.t_sched_train, self.t_sched_infer
        self.t_sched_infer = 1
        _, sigma, _ = self.get_noise_info([self.num_noises] + list(x.size()[1:]))
        self.t_sched_train, self.t_sched_infer = tmp1, tmp2
        distances = torch.sort(distances, descending=False).values
        threshold = distances[int(qtl * len(distances))] / 2
        pc = (sigma > threshold).float().mean()
        print(f"Estimated pc = {pc:.2f}")
        return pc.item()

    @torch.inference_mode()
    def estimate_null_loss(self, dl, num_estimates=200):
        # Create initial (null) model
        path = "--".join(self.method_path.split("--")[:-1] + ["pre"])
        conf = OmegaConf.load(os.path.join(path, "configuration.yaml"))
        model_module = importlib.import_module("models." + conf.model.name)
        with self.fabric.init_module():
            model = model_module.Model(conf.model, dl.dataset.meta)
        model = self.fabric.setup(model)
        model.eval()
        # Iterate some batches
        losses = []
        for i, batch in enumerate(self.progbar(dl, desc="Null loss")):
            _, x, y = batch[:3]
            n, s = self.get_noise_info(x.size(), training=True)
            xhat = self.get_model_pred(model, x, n, s, y)
            l = self.get_model_loss(x, xhat, s)
            losses.append(l)
            if i + 1 >= num_estimates:
                break
        losses = torch.stack(losses, dim=-1)
        # Get the average
        loss_null = losses.mean()
        print(f"Estimated L_null = {loss_null:.3f}")
        return loss_null.item()


###########################################################################
###########################################################################


@torch.inference_mode()
def get_topk(topk, all_idx, scores):
    # Sort and get correct indices
    ids = torch.argsort(scores, dim=-1, descending=True)
    ids = ids[:, :topk]
    ret_idx = []
    for i in range(len(ids)):
        ret_idx.append(all_idx[ids[i]])
    ret_idx = torch.stack(ret_idx, dim=0)
    return ret_idx


###########################################################################
###########################################################################
