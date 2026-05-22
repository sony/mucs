import sys
import torch, math
from einops import rearrange, repeat

from lib import posemb
from utils import pytorch_utils

###############################################################################


class PatchEmbedder(torch.nn.Module):

    def __init__(
        self,
        patch_size,
        image_size,
        out_cha,
        norm=None,
        bias=True,
    ):
        super().__init__()
        in_cha, height, width = image_size
        # Pad
        pad_h = (patch_size[0] - height % patch_size[0]) % patch_size[0]
        pad_w = (patch_size[1] - width % patch_size[1]) % patch_size[1]
        self.pad = torch.nn.ZeroPad2d((0, pad_w, 0, pad_h))
        self.grid_size = (
            (height + pad_h) // patch_size[0],
            (width + pad_w) // patch_size[1],
        )
        # Projection
        self.proj = torch.nn.Conv2d(
            in_cha, out_cha, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        # torch.nn.init.xavier_uniform_(
        #     self.proj.weight.view([self.proj.weight.size(0), -1])
        # )
        # if self.proj.bias is not None:
        #     torch.nn.init.constant_(self.proj.bias, 0)
        # Norm
        if norm is None or norm.lower() == "none":
            self.norm = torch.nn.Identity()
        elif norm.lower() == "layer_affine":
            self.norm = torch.nn.LayerNorm(out_cha, elementwise_affine=True)
        elif norm.lower() == "layer_noaffine":
            self.norm = torch.nn.LayerNorm(out_cha, elementwise_affine=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.pad(x)
        x = self.proj(x)
        x = rearrange(x, "b c gh gw -> b (gh gw) c")
        x = self.norm(x)
        return x


class PatchRecoverer(torch.nn.Module):

    def __init__(
        self, num_cha, patch_size, grid_size, latent_size, bias=True, zero_init=False
    ):
        super().__init__()
        self.grid_size = grid_size
        self.latent_size = latent_size
        self.proj = torch.nn.ConvTranspose2d(
            num_cha,
            self.latent_size[0],
            patch_size,
            stride=patch_size,
            bias=bias,
        )
        if zero_init:
            torch.nn.init.constant_(self.proj.weight, 0)
            if self.proj.bias is not None:
                torch.nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = rearrange(
            x, "b (gh gw) c -> b c gh gw", gh=self.grid_size[0], gw=self.grid_size[1]
        ).contiguous()
        x = self.proj(x)
        x = x[:, :, : self.latent_size[1], : self.latent_size[2]]
        return x


###############################################################################


class PosEncoder(torch.nn.Module):

    def __init__(self, embtype, num_cha, grid_size, cla_token=False, extra_tokens=0):
        super().__init__()
        if embtype == "random":
            emb = posemb.get_random(num_cha, grid_size)
            self.emb = torch.nn.Parameter(emb.unsqueeze(0))
        elif embtype == "sincos2d":
            emb = posemb.get_2d_sincos_pos_embed(num_cha, grid_size)
            self.register_buffer("emb", emb.unsqueeze(0))
        else:
            raise NotImplementedError
        if cla_token:
            self.cla = torch.nn.Parameter(torch.zeros(1, num_cha))
        else:
            self.cla = None
        if extra_tokens > 0:
            self.extra = torch.nn.Parameter(torch.zeros(extra_tokens, num_cha))
        else:
            self.extra = None

    def forward(self, x):
        assert x.size(1) <= self.emb.size(1)
        x = x + self.emb
        if self.extra is not None:
            extra = repeat(self.extra, "l c -> b l c", b=x.size(0))
            x = torch.cat([extra, x], dim=1)
        if self.cla is not None:
            cla = repeat(self.cla, "l c -> b l c", b=x.size(0))
            x = torch.cat([cla, x], dim=1)
        return x


###############################################################################


class SigmaEmbedder(torch.nn.Module):

    def __init__(self, num_cha, num_fourier_feats=256, bias=True):
        super().__init__()
        assert num_fourier_feats % 2 == 0
        # Frequencies
        freq = torch.linspace(0, 1, num_fourier_feats // 2 + 1)[1:]
        freq = (freq.exp() - 1) / (math.exp(1) - 1)
        self.register_buffer("freq", 2 * torch.pi * freq)
        # MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_fourier_feats, num_cha, bias=bias),
            torch.nn.SiLU(),
            torch.nn.Linear(num_cha, num_cha, bias=bias),
        )
        """
        torch.nn.init.normal_(self.mlp[0].weight, std=0.02)
        if self.mlp[0].bias is not None:
            torch.nn.init.constant_(self.mlp[0].bias, 0)
        torch.nn.init.normal_(self.mlp[2].weight, std=0.02)
        if self.mlp[2].bias is not None:
            torch.nn.init.constant_(self.mlp[2].bias, 0)
        #"""

    def forward(self, x):
        assert x.ndim == 1
        x = x.view(-1, 1) * self.freq.view(1, -1)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = self.mlp(x)
        x = rearrange(x, "b c -> b 1 c")
        return x


###############################################################################


class LabelEmbedder(torch.nn.Module):

    def __init__(self, num_classes, num_cha, std=1, possible_drop=False):
        super().__init__()
        self.num_classes = num_classes
        self.possible_drop = possible_drop
        self.emb = torch.nn.Embedding(
            self.num_classes + (1 if self.possible_drop else 0),
            num_cha,
        )
        # torch.nn.init.normal_(self.emb.weight, std=std)

    def forward(self, x, p_drop=0):
        assert x.ndim == 1
        if self.possible_drop:
            m = torch.rand(x.size(), device=x.device) < p_drop
            x = torch.where(m | (x < 0), self.num_classes, x)
        x = self.emb(x)
        x = rearrange(x, "b c -> b 1 c")
        return x


###############################################################################


class VectorEmbedder(torch.nn.Module):

    def __init__(self, dim_in, num_cha, std=1, possible_drop=False, eps=1e-8):
        super().__init__()
        self.possible_drop = possible_drop
        self.eps = eps
        if self.possible_drop:
            self.xdrop = torch.nn.Parameter(torch.zeros(dim_in))
        self.emb = torch.nn.Linear(dim_in, num_cha, bias=False)
        # torch.nn.init.normal_(self.emb.weight, std=std)

    def forward(self, x, p_drop=0):
        assert x.ndim == 2
        if self.possible_drop:
            m = torch.rand(len(x), 1, device=x.device) < p_drop
            zeronorm = x.pow(2).sum(-1, keepdim=True) < self.eps
            xdrop = repeat(self.xdrop, "d -> b d", b=len(x))
            x = torch.where(m | zeronorm, xdrop, x)
        x = self.emb(x)
        x = rearrange(x, "b c -> b 1 c")
        return x


class NewVectorEmbedder(torch.nn.Module):

    def __init__(
        self,
        dim_in,
        num_cha,
        unconditional=False,
        possible_drop=False,
        p_drop_mlp=0,
        eps=1e-8,
    ):
        super().__init__()
        self.unconditional = unconditional
        if not unconditional:
            self.possible_drop = possible_drop
            self.eps = eps
            if self.possible_drop:
                self.xdrop = torch.nn.Parameter(torch.zeros(dim_in))
            self.emb_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim_in, num_cha),
                torch.nn.SiLU(),
                torch.nn.Dropout(p_drop_mlp),
                torch.nn.Linear(num_cha, num_cha),
            )

    def forward(self, x, p_drop=0):
        assert x.ndim == 2
        if self.unconditional:
            return 0
        else:
            if self.possible_drop:
                m = torch.rand(len(x), 1, device=x.device) < p_drop
                zeronorm = x.pow(2).sum(-1, keepdim=True) < self.eps
                xdrop = repeat(self.xdrop, "d -> b d", b=len(x))
                x = torch.where(m | zeronorm, xdrop, x)
            x = self.emb_mlp(x)
            x = rearrange(x, "b c -> b 1 c")
        return x


###############################################################################


class DiTBlock(torch.nn.Module):
    # Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

    def __init__(
        self, num_cha, num_heads, mlp_ratio, num_cha_cond=None, bias=True, p_drop=0
    ):
        super().__init__()
        if num_cha_cond is None:
            num_cha_cond = num_cha
        num_hidden = int(num_cha * mlp_ratio)
        self.norm1 = torch.nn.LayerNorm(num_cha, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(
            num_cha,
            num_heads,
            p_drop=p_drop,
            proj_bias=bias,
        )
        self.norm2 = torch.nn.LayerNorm(num_cha, elementwise_affine=False, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_cha, num_hidden, bias=bias),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(num_hidden, num_cha, bias=bias),
        )
        """
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        #"""
        self.adapter = ConditionAdapter(num_cha_cond, num_cha, expand=6, bias=bias)

    def forward(self, x, c):
        g1, s1, b1, g2, s2, b2 = self.adapter(c).chunk(6, dim=-1)
        x = x + g1 * self.attention((1 + s1) * self.norm1(x) + b1)
        x = x + g2 * self.mlp((1 + s2) * self.norm2(x) + b2)
        return x


class DiTFinal(torch.nn.Module):
    # Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

    def __init__(self, num_cha, num_cha_cond=None, bias=True, p_drop=0):
        super().__init__()
        if num_cha_cond is None:
            num_cha_cond = num_cha
        self.norm = torch.nn.LayerNorm(num_cha, elementwise_affine=False, eps=1e-6)
        self.adapter = ConditionAdapter(num_cha_cond, num_cha, expand=2, bias=bias)

    def forward(self, x, c):
        s, b = self.adapter(c).chunk(2, dim=-1)
        x = (1 + s) * self.norm(x) + b
        return x


class ConditionAdapter(torch.nn.Module):

    def __init__(self, num_in, num_out, expand=2, bias=True, zero_init=True):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(num_in, expand * num_out, bias=bias),
        )
        if zero_init:
            torch.nn.init.constant_(self.adapter[-1].weight, 0)
            if self.adapter[-1].bias is not None:
                torch.nn.init.constant_(self.adapter[-1].bias, 0)

    def forward(self, x):
        return self.adapter(x)


###############################################################################


class SelfAttention(torch.nn.Module):
    # Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_norm=False,
        proj_bias=True,
        p_drop=0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.p_drop = p_drop

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm:
            self.q_norm = torch.nn.RMSNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = torch.nn.RMSNorm(self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
        self.drop = torch.nn.Dropout(self.p_drop)
        self.proj = torch.nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.p_drop if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.drop(x)
        x = self.proj(x)
        return x


###############################################################################


class ConvBlock2d(torch.nn.Module):

    def __init__(
        self,
        in_cha,
        out_cha,
        resample,
        kern,
        down=True,
        hid_cha_factor=2,
        normtype="instance",
        bias=True,
        zero_init=True,
    ):
        super().__init__()
        assert kern % 2 == 1
        mid_cha = int(hid_cha_factor * max(in_cha, out_cha))
        if normtype is None or normtype == "none":
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()
        elif normtype == "batch":
            self.norm1 = torch.nn.BatchNorm2d(in_cha)
            self.norm2 = torch.nn.BatchNorm2d(mid_cha)
        elif normtype == "instance":
            self.norm1 = torch.nn.InstanceNorm2d(in_cha, affine=True)
            self.norm2 = torch.nn.InstanceNorm2d(mid_cha, affine=True)
        elif normtype == "layer":
            self.norm1 = torch.nn.GroupNorm(1, in_cha)
            self.norm2 = torch.nn.GroupNorm(1, mid_cha)
        else:
            raise NotImplementedError
        self.act = torch.nn.SiLU()
        if down:
            self.conv1 = torch.nn.Conv2d(
                in_cha, mid_cha, resample, stride=resample, bias=bias
            )
        else:
            self.conv1 = torch.nn.ConvTranspose2d(
                in_cha, mid_cha, resample, stride=resample, bias=bias
            )
        self.conv2 = torch.nn.Conv2d(
            mid_cha, out_cha, kern, padding=kern // 2, bias=bias
        )
        if zero_init:
            torch.nn.init.constant_(self.conv2.weight, 0)
            if self.conv2.bias is not None:
                torch.nn.init.constant_(self.conv2.bias, 0)
        if out_cha != in_cha or resample != 1:
            if down:
                self.skip = torch.nn.Conv2d(
                    in_cha, out_cha, resample, stride=resample, bias=bias
                )
            else:
                self.skip = torch.nn.ConvTranspose2d(
                    in_cha, out_cha, resample, stride=resample, bias=bias
                )
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x))) + r
        return x


###############################################################################


def get_Conv2dEncDec(
    in_cha,
    channels,
    resample,
    kern,
    last_factor=1,
    normtype="instance",
    bn_enc=False,
    bn_dec=False,
):
    assert len(channels) == len(resample)
    if len(channels) == 0:
        return torch.nn.Identity(), torch.nn.Identity()
    args = []
    c1 = in_cha
    for c2, r in zip(channels, resample):
        args.append([c1, c2, r])
        c1 = c2
    blocks = []
    for c1, c2, r in args:
        blocks.append(ConvBlock2d(c1, c2, r, kern, down=True, normtype=normtype))
    if bn_enc:
        blocks.append(torch.nn.BatchNorm2d(c2))
    encoder = torch.nn.Sequential(*blocks)
    blocks = []
    if bn_dec:
        blocks.append(torch.nn.BatchNorm2d(c2))
    args[0][0] *= last_factor
    for c1, c2, r in args[::-1]:
        blocks.append(ConvBlock2d(c2, c1, r, kern, down=False, normtype=normtype))
    decoder = torch.nn.Sequential(*blocks)
    return encoder, decoder


def get_Conv2dEncDecU(
    in_cha,
    channels,
    resample,
    kern,
    last_factor=1,
    normtype="none",
    bn_enc=False,
    bn_dec=False,
):
    assert len(channels) == len(resample)
    if len(channels) == 0:
        return torch.nn.Identity(), torch.nn.Identity()
    args = []
    c1 = in_cha
    for c2, r in zip(channels, resample):
        args.append([c1, c2, r])
        c1 = c2
    blocks = []
    for c1, c2, r in args:
        blocks.append(ConvBlock2d(c1, c2, r, kern, down=True, normtype=normtype))
    if bn_enc:
        blocks.append(torch.nn.BatchNorm2d(c2))
    encoder = torch.nn.ModuleList(blocks)
    blocks = []
    if bn_dec:
        blocks.append(torch.nn.BatchNorm2d(c2))
    args[0][0] *= last_factor
    for c1, c2, r in args[::-1]:
        blocks.append(ConvBlock2d(2 * c2, c1, r, kern, down=False, normtype=normtype))
    decoder = torch.nn.ModuleList(blocks)
    return encoder, decoder


###############################################################################


class Linear(torch.nn.Module):

    def __init__(self, in_cha, out_cha, dim=1, bias=True):
        super().__init__()
        self.dim = dim
        self.proj = torch.nn.Linear(in_cha, out_cha, bias=bias)

    def forward(self, x):
        x = self.proj(x.transpose(self.dim, -1)).transpose(-1, self.dim)
        return x


###############################################################################


class TrafoBlock(torch.nn.Module):

    def __init__(self, num_cha, num_heads, mlp_ratio, bias=True, p_drop=0):
        super().__init__()
        num_hidden = int(num_cha * mlp_ratio)
        self.norm1 = torch.nn.LayerNorm(num_cha, elementwise_affine=True)
        self.attention = SelfAttention(
            num_cha,
            num_heads,
            p_drop=p_drop,
            proj_bias=bias,
        )
        self.norm2 = torch.nn.LayerNorm(num_cha, elementwise_affine=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_cha, num_hidden, bias=bias),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(num_hidden, num_cha, bias=bias),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


###############################################################################


class GeMPool(torch.nn.Module):

    def __init__(self, ncha=1, init=3, eps=1e-6):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        pinit = math.log(math.exp(init - 1) - 1)
        self.p = torch.nn.Parameter(pinit * torch.ones(1, 1, ncha))
        self.eps = eps

    def forward(self, h):
        # b,t,c
        pow = 1 + self.softplus(self.p)
        h = h.clamp(min=self.eps).pow(pow)
        h = h.mean(1).pow(1 / pow[:, 0, :])
        return h


###############################################################################
