import sys
import torch, math
from einops import rearrange, repeat

from lib import nn
from utils import pytorch_utils

###############################################################################


class Model(torch.nn.Module):

    ######################################################################

    def __init__(self, conf, metadata):
        super().__init__()
        self.args = (conf, metadata)
        self.unconditional = conf.arch.unconditional
        # Variables
        self.p_drop_c = conf.loss.p_drop_cond
        self.image_size = list(metadata["image_size"])
        self.cond_dim = metadata["label_dim"]
        latent_size = [
            conf.arch.frontend.channels[-1],
            self.image_size[1] // math.prod(conf.arch.frontend.resample),
            self.image_size[2] // math.prod(conf.arch.frontend.resample),
        ]
        # Encoder/decoder
        self.encoder, self.decoder = nn.get_Conv2dEncDec(
            self.image_size[0],
            conf.arch.frontend.channels,
            conf.arch.frontend.resample,
            conf.arch.frontend.kern,
            last_factor=conf.arch.frontend.last_factor,
        )
        self.last = nn.Linear(
            self.image_size[0] * conf.arch.frontend.last_factor,
            self.image_size[0],
            bias=False,
        )
        torch.nn.init.constant_(self.last.proj.weight, 0)
        # Embeddings
        self.x_embedder = nn.PatchEmbedder(
            conf.arch.dit.patch_size,
            latent_size,
            conf.arch.dit.num_channels,
        )
        self.pos_enc = nn.PosEncoder(
            "sincos2d",
            conf.arch.dit.num_channels,
            self.x_embedder.grid_size,
            extra_tokens=conf.arch.dit.extra_tokens,
        )
        self.extra_tokens = conf.arch.dit.extra_tokens
        self.s_embedder = nn.SigmaEmbedder(
            conf.arch.dit.num_channels_cond,
        )
        self.y_embedder = nn.NewVectorEmbedder(
            self.cond_dim,
            conf.arch.dit.num_channels_cond,
            unconditional=conf.arch.unconditional,
            possible_drop=True,
            p_drop_mlp=conf.arch.dit.p_drop,
        )
        self.a_embedder = nn.LabelEmbedder(
            2,
            conf.arch.dit.num_channels_cond,
            possible_drop=False,
        )
        # DiT backbone
        blocks = []
        for _ in range(conf.arch.dit.num_blocks):
            blocks.append(
                nn.DiTBlock(
                    conf.arch.dit.num_channels,
                    conf.arch.dit.num_heads,
                    conf.arch.dit.mlp_ratio,
                    num_cha_cond=conf.arch.dit.num_channels_cond,
                    p_drop=conf.arch.dit.p_drop,
                )
            )
        self.dit_backbone = torch.nn.ModuleList(blocks)
        self.dit_final = nn.DiTFinal(
            conf.arch.dit.num_channels,
            num_cha_cond=conf.arch.dit.num_channels_cond,
            p_drop=conf.arch.dit.p_drop,
        )
        # Unpatchify
        self.unpatchify = nn.PatchRecoverer(
            conf.arch.dit.num_channels,
            conf.arch.dit.patch_size,
            self.x_embedder.grid_size,
            latent_size,
        )
        # Loss
        self.sigma_P_mean, self.sigma_P_std = conf.loss.sigma_P
        self.sigma_data = conf.loss.sigma_data
        # Sampler
        self.sampler = conf.sampler

    ######################################################################

    def forward(self, x, sigma, y=None, a=None):
        # Prepare
        assert sigma.ndim == 1
        if y is None:
            y = torch.zeros(
                len(x), self.cond_dim, dtype=x.dtype, device=x.device
            )  # no cond
        else:
            assert y.ndim == 2
        if a is None:
            a = torch.zeros_like(sigma).long()  # no aug
        else:
            assert a.ndim == 1
        # Precond
        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = (1e-6 + sigma).log() / 4
        # Forward F_x and create D_x
        F_x = self._network(c_in * x, c_noise.view(-1), y, a)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def _network(self, x, s, y, a):
        # Encoder
        x = self.encoder(x)
        # Get embeddings
        x = self.pos_enc(self.x_embedder(x))
        c = (
            self.s_embedder(s)
            + self.y_embedder(y, p_drop=self.p_drop_c if self.training else 0)
            + self.a_embedder(a)
        )
        # Backbone
        for block in self.dit_backbone:
            x = block(x, c)
        x = self.dit_final(x, c)
        # Output
        x = x[:, self.extra_tokens :, :]
        x = self.unpatchify(x)
        # Decoder
        x = self.decoder(x)
        x = self.last(x)
        return x

    ######################################################################

    def loss(self, x, y=None, a=None, idx=None, rng=None):
        # Noise
        if rng is None:
            s = torch.randn([len(x), 1, 1, 1], device=x.device)
            sigma = (self.sigma_P_std * s + self.sigma_P_mean).exp()
            n = sigma * torch.randn_like(x)
        else:
            s = rng.randn([len(x), 1, 1, 1])
            sigma = (self.sigma_P_std * s + self.sigma_P_mean).exp()
            n = sigma * rng.randn_like(x)
        # Forward
        xhat = self(x + n, sigma.view(-1), y=y, a=a)
        # Loss
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        loss = (weight * (xhat - x).pow(2)).mean()
        # Logging
        logdict = {
            "l_main": loss,
        }
        return loss, logdict

    ######################################################################

    @torch.inference_mode()
    def generate(
        self,
        y=None,
        rng=None,
        num_samples=1,
        num_steps=None,
        cfg=None,
        sigma_min=None,
        sigma_max=None,
        rho=None,
        S_churn=None,
        S_min=None,
        S_max=None,
        S_noise=None,
        return_extra=False,
    ):
        # Adapted from https://github.com/NVlabs/edm/blob/main/generate.py
        # Setup params
        num_steps = self.sampler.num_steps if num_steps is None else num_steps
        sigma_min = (
            self.sampler.sigma_min
            if sigma_min is None
            else max(sigma_min, self.sampler.sigma_min)
        )
        sigma_max = (
            self.sampler.sigma_max
            if sigma_max is None
            else min(sigma_max, self.sampler.sigma_max)
        )
        cfg = self.sampler.cfg if cfg is None else cfg
        rho = self.sampler.rho if rho is None else rho
        S_churn = self.sampler.S_churn if S_churn is None else S_churn
        S_min = self.sampler.S_min if S_min is None else S_min
        S_max = self.sampler.S_max if S_max is None else S_max
        S_noise = self.sampler.S_noise if S_noise is None else S_noise
        # Prepare
        if y is None:
            y = torch.zeros(1, 1, dtype=torch.float, device=self.last.device)  # no cond
            cfg = 1
        else:
            assert y.ndim == 2
        if num_samples > 1:
            y = repeat(y, "b -> (b n)", n=num_samples)
        if rng is None:
            rng = pytorch_utils.StackedRandomGenerator(num=len(y), device=y.device)

        def denoise(x, t):
            t = torch.as_tensor([t] * len(x), device=x.device)
            Dx = self(x, t, y=y)
            if cfg == 1:
                return Dx
            return cfg * Dx + (1 - cfg) * self(x, t)

        # Time step discretization
        step_indices = torch.arange(num_steps, device=y.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N=0
        # Main sampling loop
        x_next = t_steps[0] * rng.randn([len(y)] + self.image_size)
        if return_extra:
            noised1, noised2 = [], []
            expect1, expect2 = [], []
            score1, score2 = [], []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            # Increase noise temporarily
            if S_churn > 0 and S_min <= t_cur <= S_max:
                gamma = min(S_churn / num_steps, math.sqrt(2) - 1)
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (
                    t_hat**2 - t_cur**2
                ).sqrt() * S_noise * rng.randn_like(x_cur)
            else:
                t_hat = t_cur
                x_hat = x_cur
            # Euler step (with CFG)
            d_phi = denoise(x_hat, t_hat)
            d_cur = (x_hat - d_phi) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if return_extra:
                noised1.append(x_hat.cpu())
                expect1.append(d_phi.cpu())
                score1.append(((t_next - t_hat) * d_cur).cpu())
            # 2nd order correction
            if i < num_steps - 1:
                d_phi = denoise(x_next, t_next)
                d_prime = (x_next - d_phi) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                if return_extra:
                    noised2.append(x_next.cpu())
                    expect2.append(d_phi.cpu())
                    score2.append(
                        ((t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)).cpu()
                    )
        if return_extra:
            extra = {
                "t_steps": repeat(
                    t_steps.unsqueeze(0), "1 n -> b n", b=len(x_next)
                ).cpu(),
                "noised1": torch.stack(noised1, dim=1),
                "noised2": torch.stack(noised2, dim=1),
                "expect1": torch.stack(expect1, dim=1),
                "expect2": torch.stack(expect2, dim=1),
                "score1": torch.stack(score1, dim=1),
                "score2": torch.stack(score2, dim=1),
            }
            return x_next, extra
        return x_next


###############################################################################
