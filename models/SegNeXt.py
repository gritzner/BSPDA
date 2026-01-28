# https://arxiv.org/abs/2209.08575
import core
import numpy as np
import torch
import torch.nn as nn
import models


class SegNeXtMSCABlock(nn.Module):
    def __init__(self, channels, use_bias):
        super().__init__()
        
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=use_bias)
        self.branch0 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=use_bias),
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=use_bias)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels, bias=use_bias),
            nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels, bias=use_bias)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels, bias=use_bias),
            nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels, bias=use_bias)
        )
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=use_bias)
        
    def forward(self, x):
        z = self.conv0(x)
        return x * self.conv1(z + self.branch0(z) + self.branch1(z) + self.branch2(z))


class SegNeXtMSCANBlock(nn.Module):
    def __init__(self, channels, expansion_factor, activation, normalization, use_bias):
        super().__init__()
        
        self.norm = normalization(channels, affine=use_bias)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=use_bias),
            activation(None),
            SegNeXtMSCABlock(channels, use_bias),
            nn.Conv2d(channels, channels, 1, bias=use_bias)
        )
        self.att_scaling = nn.Parameter(.01 * torch.ones((1, channels, 1, 1), dtype=torch.float32), requires_grad=True)
        
        hidden_channels = int(channels * expansion_factor)
        self.mlp = nn.Sequential(
            normalization(channels, affine=use_bias),
            nn.Conv2d(channels, hidden_channels, 1, bias=use_bias),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=use_bias),
            activation(None),
            nn.Conv2d(hidden_channels, channels, 1, bias=use_bias)
        )
        self.mlp_scaling = nn.Parameter(.01 * torch.ones((1, channels, 1, 1), dtype=torch.float32), requires_grad=True)
        
    def forward(self, x):
        z = self.norm(x)
        x = x + self.att_scaling * (z + self.attention(z))
        x = x + self.mlp_scaling * self.mlp(x)
        return x

class SegNeXtEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, num_blocks, expansion_factor, parent):
        super().__init__()
        self.num_blocks = num_blocks
        
        self.embed_patch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2 if downsample else 1, padding=1, bias=parent.use_bias),
            parent.normalization(out_channels, affine=parent.use_bias)
        )
        for i in range(self.num_blocks):
            setattr(self, f"block{i}", SegNeXtMSCANBlock(
                out_channels, expansion_factor, parent.activation, parent.normalization, parent.use_bias
            ))
        if parent.self_normalizing:
            self.norm = parent.normalization(out_channels, elementwise_affine=parent.use_bias)
        else:
            self.norm = nn.LayerNorm(out_channels, elementwise_affine=parent.use_bias)
        
    def forward(self, x):
        x = self.embed_patch(x)
        for i in range(self.num_blocks):
            x = getattr(self, f"block{i}")(x)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SegNeXtFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        
        self.downsampling = config.downsampling
        self.num_features, num_blocks = {
            "tiny": ((16, 32, 64, 160, 256), (3, 3, 5, 2)),
            "small": ((32, 64, 128, 320, 512), (2, 2, 4, 2)),
            "base": ((32, 64, 128, 320, 512), (3, 3, 12, 3)),
            "large": ((32, 64, 128, 320, 512), (3, 5, 27, 3)),
        }[config.size]
        self.use_bias = getattr(config, "use_bias", True)
        self.activation = lambda _: nn.GELU()
        self.padding = nn.ZeroPad2d
        self.normalization = nn.BatchNorm2d
        self.dropout = config.dropout
        assert len(self.dropout.ps) == 5
        self.dropout.func = nn.functional.dropout2d if self.dropout.channelwise else nn.functional.dropout

        self.self_normalizing = getattr(config, "self_normalizing", False)
        if self.self_normalizing:
            self.activation = lambda _: nn.SELU()
            self.normalization = nn.Identity
            self.dropout.func = nn.functional.alpha_dropout
        
        assert self.downsampling in (2, 3, 4, 5)
        
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, self.num_features[0], 3, stride=2, padding=1, bias=self.use_bias),
            self.normalization(self.num_features[0], affine=self.use_bias),
            self.activation(None)
        )
        for i in range(1, 5):
            setattr(self, f"stage{i}", SegNeXtEncoderStage(
                self.num_features[i-1],
                self.num_features[i],
                self.downsampling > i,
                num_blocks[i-1],
                8 if i < 3 else 4,
                self
            ))
        self.to(core.device)
        
        if getattr(config, "pretrained", False):
            print("using pretrained encoder weights")
            with core.open(core.user.model_weights_paths[f"SegNeXt_{config.size}"], "rb") as f:
                weights = torch.load(f, map_location=core.device)

            if self.use_bias:
                w = self.state_dict()["stage0.0.bias"]
                if self.self_normalizing:
                    nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
                weights["stage0.0.bias"] = w
            w = self.state_dict()["stage0.0.weight"]
            if self.self_normalizing:
                nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
            weights["stage0.0.weight"] = w
            
            for k in set(weights.keys()) - set(self.state_dict().keys()):
                del weights[k]
            
            self.load_state_dict(weights)
        elif self.self_normalizing:
            weights = self.state_dict()
            for k, w in weights.items():
                nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
                weights[k] = w
            self.load_state_dict(weights)
    
    def forward(self, x):
        y = [x]
        for i, p in enumerate(self.dropout.ps):
            y.append(self.dropout.func(
                getattr(self, f"stage{i}")(y[-1]), p, self.training
            ))
        return y[1:]


class SegNeXt(nn.Module):
    def __init__(self, config, params):
        super().__init__()

        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for SegNeXt must be square")
        
        self.encoder = models.feature_extractor(config.encoder, params.input_shape[0])
        assert self.encoder.downsampling == 5
        activation = lambda _: nn.ReLU()
        normalization = nn.GroupNorm
        if self.encoder.self_normalizing:
            activation = self.encoder.activation
            normalization = self.encoder.normalization
        use_bias = getattr(config, "use_bias", self.encoder.use_bias)
        self.model_name = f"SegNeXt-{config.encoder.size}"
        
        hamburger_channels, hamburger_MD_R = {
            "tiny": (256, 16), "small": (256, 16), "base": (512, 64), "large": (1024, 64)
        }[config.encoder.size]
        
        self.lower_bread0 = nn.Sequential(
            nn.Conv2d(np.sum(self.encoder.num_features[-3:]), hamburger_channels, 1, bias=use_bias),
            normalization(32, hamburger_channels, affine=use_bias),
            activation(None)
        )
        self.lower_bread1 = nn.Sequential(
            nn.Conv2d(hamburger_channels, hamburger_channels, 1, bias=use_bias),
            activation(None)
        )
        self.ham = NMF2D({"MD_R": hamburger_MD_R})
        self.upper_bread0 = nn.Sequential(
            nn.Conv2d(hamburger_channels, hamburger_channels, 1, bias=use_bias),
            normalization(32, hamburger_channels, affine=use_bias)
        )
        self.upper_bread1 = nn.Sequential(
            nn.Conv2d(hamburger_channels, hamburger_channels, 1, bias=use_bias),
            normalization(32, hamburger_channels, affine=use_bias),
            activation(None)
        )
        self.classifier = nn.Conv2d(hamburger_channels, params.num_classes, 1, bias=use_bias)
        
        if self.encoder.self_normalizing:
            weights = self.state_dict()
            for k, w in weights.items():
                if k[:8] == "encoder.":
                    continue
                nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
                weights[k] = w
            self.load_state_dict(weights)
        
        self.to(core.device)

    def forward(self, x, yt=None, ce_loss_func=None, weight=None, ignore_index=None):
        y = self.encoder(x)
        y = [
            nn.functional.interpolate(y, scale_factor=2**(3+i), mode="bilinear", align_corners=False) for i, y in enumerate(y[2:])
        ]
        y = self.lower_bread0(torch.cat(y, dim=1))
        y = nn.functional.relu(y + self.upper_bread0(self.ham(self.lower_bread1(y))))
        y = self.classifier(self.upper_bread1(y))
        if yt is None:
            return y
        return y, ce_loss_func(y, yt, weight=weight, ignore_index=ignore_index)



# === the code below this comment is taken from mmseg/models/decode_heads/ham_head.py of commit c87bcae of ===
# === https://github.com/Visual-Attention-Network/SegNeXt                                                  ===
# ===                                                                                                      ===
# === it has been slightly modified to remove unwanted writes to stdout and be compatible with the other   ===
# === imports of this file                                                                                 ===

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        #print('spatial', self.spatial)
        #print('S', self.S)
        #print('D', self.D)
        #print('R', self.R)
        #print('train_steps', self.train_steps)
        #print('eval_steps', self.eval_steps)
        #print('inv_t', self.inv_t)
        #print('eta', self.eta)
        #print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = torch.nn.functional.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = torch.nn.functional.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef
