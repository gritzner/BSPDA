# https://timm.fast.ai/
import core
import models
import utils
import numpy as np
import torch
import torch.nn as nn
import timm


class TIMMFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        # timm.list_models(pretrained=True)
        self.model = timm.create_model(
            config.backbone,
            pretrained = config.pretrained,
            features_only = True,
            in_chans = in_channels
        )
        assert len(self.model.feature_info) <= 5
        self.hierarchy_map = [None] * 5
        for i, fi in enumerate(self.model.feature_info):
            j = int(round(np.log2(fi["reduction"]))) - 1
            assert self.hierarchy_map[j] is None
            self.hierarchy_map[j] = i
        assert any([not (i is None) for i in self.hierarchy_map])
        for i, j in enumerate(self.hierarchy_map):
            if not (j is None):
                continue
            j = i - 1
            while j >= 0 and self.hiearchy_map[j] is None:
                j -= 1
            if j >= 0:
                self.hiearchy_map[i] = -5 + j
                continue
            # did not find a higher resolution feature map to downsample from, therefore find a lower resolution to upsample from (must exists or the previous assertion would have failed)
            j = i + 1
            while self.hierarchy_map[j] is None:
                j += 1
            self.hierarchy_map[i] = -5 + j
        self.hierarchy_map = tuple(self.hierarchy_map)
        assert all([not (i is None) for i in self.hierarchy_map])
        self.downsampling = config.downsampling
        assert self.downsampling in set(range(1, 6))
        self.resample_all_stages = config.resample_all_stages
        self.concatenate_stages = tuple(sorted(config.concatenate_stages))
        assert len(self.concatenate_stages) < 4
        self.num_features = [fi["num_chs"] for fi in self.model.feature_info]
        self.num_features = [self.num_features[i] if i >= 0 else None for i in self.hierarchy_map]
        self.num_features = [self.num_features[5+self.hierarchy_map[i]] if j is None else j for i, j in enumerate(self.num_features)]
        for i in self.concatenate_stages:
            assert i >= 0 and i < 4
            self.num_features[-1] += self.num_features[i]
        self.num_features = tuple(self.num_features)
        self.use_bias = config.use_bias
        self.activation = utils.relu_wrapper(config.activation)
        self.padding = nn.ZeroPad2d
        self.normalization = utils.norm_wrapper(config.normalization)
        self.dropout = config.dropout
        self.dropout.func = nn.functional.dropout2d if self.dropout.channelwise else nn.functional.dropout
        self.self_normalizing = False
        self.to(core.device)
        
    def forward(self, x):
        assert x.shape[-2] == x.shape[-1]
        z = self.model(x)
        z = [z[i] if i >= 0 else None for i in self.hierarchy_map]
        z = [z[5+self.hierarchy_map[i]] if j is None else j for i, j in enumerate(z)]
        for i, zi in enumerate(z, start=1):
            assert zi.shape[-2] == zi.shape[-1]
            target_size = x.shape[-1] // 2**min(i, self.downsampling)
            if zi.shape[-1] == target_size or (i < self.downsampling and not self.resample_all_stages):
                continue
            if zi.shape[-1] > target_size:
                z[i-1] = nn.functional.adaptive_avg_pool2d(zi, target_size)
            else:
                z[i-1] = nn.functional.interpolate(zi, size=target_size, mode="bilinear", align_corners=True)
        if len(self.concatenate_stages) > 0:
            z[-1] = torch.cat([z[i] for i in [*self.concatenate_stages, 4]], dim=1)
        return z


class TIMMViTFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        if hasattr(config, "second_backbone"):
            self.second_model = models.feature_extractor(config.second_backbone, in_channels)
            self.predict_residual = getattr(config.second_backbone, "predict_residual", False)
        if hasattr(config, "map_input_channels"):
            self.map_input_channels = torch.as_tensor(config.map_input_channels, dtype=torch.int64, device=core.device)
            in_channels = self.map_input_channels.shape[0]
        if hasattr(config, "zero_input_channels"):
            self.zero_input_channels = torch.as_tensor(config.zero_input_channels, dtype=torch.int64, device=core.device)
        if config.pretrained:
            self.model = timm.create_model(
                config.backbone,
                pretrained = True,
                in_chans = in_channels
            )
        else:
            kwargs = {}
            if hasattr(config, "override_input_size"):
                assert len(config.override_input_size) == 2
                kwargs["img_size"] = tuple(config.override_input_size)
            self.model = timm.create_model(
                config.backbone,
                pretrained = False,
                patch_size = 2**config.downsampling,
                in_chans = in_channels,
                global_pool = "",
                class_token = False,
                **kwargs
            )
        self.model.attn_pool = None
        self.model.fc_norm = None
        self.model.head_drop = None
        self.model.head = None
        self.downsampling = config.downsampling
        if hasattr(self, "second_model") and not self.predict_residual:
            self.num_features = tuple([self.model.num_features + self.second_model.num_features[-1]] * 5)
        else:
            self.num_features = tuple([self.model.num_features] * 5)
        if hasattr(self, "second_model") and self.predict_residual and self.num_features[-1] != self.second_model.num_features[-1]:
            self.mapping_layer = nn.Conv2d(self.second_model.num_features[-1], self.num_features[-1], 1)
        self.use_bias = config.use_bias
        self.activation = utils.relu_wrapper(config.activation)
        self.padding = nn.ZeroPad2d
        self.normalization = utils.norm_wrapper(config.normalization)
        self.dropout = config.dropout
        self.dropout.func = nn.functional.dropout2d if self.dropout.channelwise else nn.functional.dropout
        self.self_normalizing = False
        self.stop_gradients = getattr(config, "stop_gradients", False)
        self.to(core.device)

    def forward(self, x):
        assert x.shape[-2] == x.shape[-1]
        assert (x.shape[-1] % 2**self.downsampling) == 0
        other_z = self.second_model(x)[-1] if hasattr(self, "second_model") else None
        if hasattr(self, "map_input_channels"):
            x = x[:, self.map_input_channels]
        if hasattr(self, "zero_input_channels"):
            x[:, self.zero_input_channels] = 0
        if self.stop_gradients:
            with torch.no_grad():
                z = self.forward_features(x).detach()
        else:
            z = self.forward_features(x)
        if not (other_z is None):
            if hasattr(self, "mapping_layer"):
                other_z = self.mapping_layer(other_z)
            z = z + other_z if self.predict_residual else torch.cat((z, other_z), dim=1)
        return [z] * 5
    
    def forward_features(self, x):
        n = x.shape[-1] // 2**self.downsampling
        z = self.model.forward_features(x)[:, self.model.num_prefix_tokens:]
        k = int(round(np.sqrt(z.shape[1])))
        z = z.reshape((x.shape[0], k, k, self.model.num_features))
        z = z.permute(0, 3, 1, 2)
        if n < k:
            z = nn.functional.adaptive_avg_pool2d(z, n)
        elif n > k:
            z = nn.functional.interpolate(z, size=n, mode="bilinear", align_corners=True)
        return z
