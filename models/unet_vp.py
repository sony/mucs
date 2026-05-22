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
