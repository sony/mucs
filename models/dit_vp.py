import sys
import torch, math
import numpy as np
from einops import rearrange, repeat

from lib import nn
from utils import pytorch_utils

###############################################################################


class Model(torch.nn.Module):

    ######################################################################

    def __init__(self, conf, metadata):
        super().__init__()
        # Variables
        self.p_drop_c = conf.loss.p_drop_cond
        self.image_size = list(metadata["image_size"])
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
        self.y_embedder = nn.VectorEmbedder(
            metadata["label_dim"],
            conf.arch.dit.num_channels_cond,
            possible_drop=True,
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
        self.beta_d = conf.loss.beta_d
        self.beta_min = conf.loss.beta_min
        self.M = conf.loss.M
        self.epsilon_t = conf.loss.epsilon_t
        self.sigma_min = float(self.sigma(conf.loss.epsilon_t))
        self.sigma_max = float(self.sigma(1))
        # Sampler
        self.sampler = conf.sampler

    ######################################################################

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, y=None, a=None):
        # Prepare
        x = x.to(torch.float32)
        assert sigma.ndim == 1
        if y is None:
            y = torch.zeros(len(x), 1, dtype=x.dtype, device=x.device)  # no cond
        else:
            assert y.ndim == 2
        if a is None:
            a = torch.zeros_like(sigma).long()  # no aug
        else:
            assert a.ndim == 1
        # Precond
        sigma = sigma.to(torch.float32).view(-1, 1, 1, 1)
        c_skip = 1
        c_out = - sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)
        # Forward F_x and create D_x
        F_x = self._network(c_in * x, c_noise.view(-1), y, a).to(torch.float32)
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
            s = torch.rand([len(x), 1, 1, 1], device=x.device)
            sigma = self.sigma(1 + s * (self.epsilon_t - 1))
            n = sigma * torch.randn_like(x)
        else:
            s = rng.rand([len(x), 1, 1, 1])
            sigma = self.sigma(1 + s * (self.epsilon_t - 1))
            n = sigma * rng.randn_like(x)
        # Forward
        xhat = self(x + n, sigma.view(-1), y=y, a=a)
        # Loss
        weight = 1 / sigma ** 2
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
        epsilon_s=None,
        S_churn=None,
        S_min=None,
        S_max=None,
        S_noise=None,
        return_extra=False,
    ):
        # Adapted from https://github.com/NVlabs/edm/blob/main/generate.py
        
        # Helper functions for VP & VE noise level schedules.
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
   
        epsilon_s = self.sampler.epsilon_s if epsilon_s is None else epsilon_s

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            sigma_min = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        if sigma_max is None:
            sigma_max = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)

        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        num_steps = self.sampler.num_steps if num_steps is None else num_steps

        # Time step discretization
        step_indices = torch.arange(num_steps, device=y.device)
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)

        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)

        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)

        t_steps = sigma_inv(self.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        cfg = self.sampler.cfg if cfg is None else cfg
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

        # Main sampling loop
        t_next = t_steps[0]
        x_next = rng.randn([len(y)] + self.image_size) * (sigma(t_next) * s(t_next))
        if return_extra:
            noised1 = []
            expect1 = []
            # score1, score2 = [], []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            # Increase noise temporarily
            if S_churn > 0 and S_min <= t_cur <= S_max:
                gamma = min(S_churn / num_steps, math.sqrt(2) - 1)
            else:
                gamma = 0

            t_hat = sigma_inv(self.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * rng.randn_like(x_cur)

            # Euler step (with CFG)
            h = t_next - t_hat
            denoised = denoise(x_hat / s(t_hat), sigma(t_hat))
            d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised

            # 2nd order correction
            if i < num_steps - 1:
                x_next = x_hat + h * d_cur
                if return_extra:
                    noised1.append(x_next.cpu())
                    expect1.append(denoised.cpu())

        if return_extra:
            extra = {
                "t_steps": repeat(
                    t_steps.unsqueeze(0), "1 n -> b n", b=len(x_next)
                ).cpu(),
                "noised1": torch.stack(noised1, dim=1),
                "expect1": torch.stack(expect1, dim=1),
            }
            return x_next, extra
        return x_next


###############################################################################
