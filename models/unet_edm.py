import sys
import numpy as np
import torch, math
from einops import rearrange, repeat
from torch.nn.functional import silu

from lib import unet_nn
from utils import pytorch_utils

###############################################################################

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

class Model(torch.nn.Module):

    ######################################################################

    def __init__(self, conf, metadata):
        super().__init__()
        # Variables
        self.metadata = metadata
        self.p_drop_c = conf.loss.p_drop_cond       # label_dropout
        self.image_size = list(metadata["image_size"])
        self.in_channels = self.image_size[0]
        self.out_channels = self.image_size[0]
        self.img_resolution = self.image_size[1]
        self.augment_dim = 2

        emb_channels = conf.arch.unet.model_channels * conf.arch.unet.channel_mult_emb
        noise_channels = conf.arch.unet.model_channels * conf.arch.unet.channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=conf.arch.unet.dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=conf.arch.unet.resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = unet_nn.PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_label = unet_nn.Linear(in_features=metadata["label_dim"], out_features=noise_channels, **init) if metadata["label_dim"] else None
        self.map_augment = unet_nn.LabelEmbedder(self.augment_dim, noise_channels, possible_drop=False) if self.augment_dim else None
        self.map_layer0 = unet_nn.Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = unet_nn.Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = self.in_channels
        caux = self.in_channels
        for level, mult in enumerate(conf.arch.unet.channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = conf.arch.unet.model_channels
                self.enc[f'{res}x{res}_conv'] = unet_nn.Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = unet_nn.UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
        
            for idx in range(conf.arch.unet.num_blocks):
                cin = cout
                cout = conf.arch.unet.model_channels * mult
                attn = (res in conf.arch.unet.attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = unet_nn.UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(conf.arch.unet.channel_mult))):
            res = self.img_resolution >> level
            if level == len(conf.arch.unet.channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = unet_nn.UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = unet_nn.UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = unet_nn.UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(conf.arch.unet.num_blocks + 1):
                cin = cout + skips.pop()
                cout = conf.arch.unet.model_channels * mult
                attn = (idx == conf.arch.unet.num_blocks and res in conf.arch.unet.attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = unet_nn.UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if level == 0:
                self.dec[f'{res}x{res}_aux_norm'] = unet_nn.GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = unet_nn.Conv2d(in_channels=cout, out_channels=self.out_channels, kernel=3, **init_zero)

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
            y = torch.zeros(len(x), self.metadata["label_dim"], dtype=x.dtype, device=x.device)  # no cond
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
        # Mapping.
        emb = self.map_noise(s)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = y
            if self.training and self.p_drop_c:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.p_drop_c).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and a is not None:
            emb = emb + self.map_augment(a)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, unet_nn.UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

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
