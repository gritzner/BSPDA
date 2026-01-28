# https://arxiv.org/abs/1505.04597
import core
import numpy as np
import torch
import torch.nn as nn
import models


class UNet(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for UNet must be square")

        self.encoder = models.feature_extractor(config.encoder, params.input_shape[0])
        assert self.encoder.downsampling == 4
        self.model_name = "UNet"
        
        for i, (in_channels, out_channels) in enumerate((
            (self.encoder.num_features[-1] + self.encoder.num_features[2], 512),
            (self.encoder.num_features[1] + 512, 256),
            (self.encoder.num_features[0] + 256, 128),
            (params.input_shape[0] + 128, 64)
        )):
            setattr(self, f"block{i}", nn.Sequential(
                self.encoder.padding(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3),
                self.encoder.normalization(out_channels),
                self.encoder.activation(out_channels),
                self.encoder.padding(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3),
                self.encoder.normalization(out_channels),
                self.encoder.activation(out_channels)
            ))
        self.classifier = nn.Conv2d(64, params.num_classes, kernel_size=1)
        self.to(core.device)
            
        if self.encoder.self_normalizing:
            weights = self.state_dict()
            for k, w in weights.items():
                if k[:8] == "encoder.":
                    continue
                nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
                weights[k] = w
            self.load_state_dict(weights)
    
    def forward(self, x, yt=None, ce_loss_func=None, weight=None, ignore_index=None):
        ys = self.encoder(x)
        y = ys[-1]
        for i, other in enumerate(reversed((x, *ys[:3]))):
            y = nn.functional.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = torch.cat([other, y], dim=1)
            y = getattr(self, f"block{i}")(y)
        y = self.classifier(y)
        if yt is None:
            return y
        return y, ce_loss_func(y, yt, weight=weight, ignore_index=ignore_index)
