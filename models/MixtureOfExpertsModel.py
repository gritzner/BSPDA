import core
import torch.nn as nn
import models


class MixtureOfExpertsModel(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        assert core.get_object_meta_info(config.base_model)[0] == "ForestEnsemble"
        self.base_model = core.create_object(
            models, config.base_model,
            input_shape = params.input_shape,
            num_classes = params.num_classes
        )
        self.gating_model = nn.Conv2d(
            self.base_model.base_model.encoder_suffix[-3].out_channels,
            self.base_model.num_extra_branches + 1,
            1, bias=config.use_bias
        )
        self.loss_weight = config.loss_weight
        assert 0 <= self.loss_weight and self.loss_weight <= 1
        self.to(core.device)
        
    def prepare_for_epoch(self, epoch, epochs):
        self.base_model.prepare_for_epoch(epoch, epochs)
        
    def forward(self, x, yt=None, ce_loss_func=None, weight=None, ignore_index=None):
        x, z = self.base_model(x, return_features=True)
        yp = self.base_model(
            x=x, yt=yt, ce_loss_func=ce_loss_func, weight=weight, ignore_index=ignore_index,
            z=z, stack=True
        )
        if not (yt is None):
            yp, loss = yp
        g = self.gating_model(z).permute(1, 0, 2, 3).unsqueeze(2)
        g = g.softmax(0)
        g = g.repeat_interleave(yp.shape[-1] // g.shape[-1], 4)
        g = g.repeat_interleave(yp.shape[-2] // g.shape[-2], 3)
        yp = (g * yp).sum(0)
        if yt is None:
            return yp
        loss = (1-self.loss_weight)*loss + self.loss_weight*ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
        return yp, loss
