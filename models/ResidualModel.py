import core
import torch
import torch.nn as nn
import models


class ResidualModel(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        assert core.get_object_meta_info(config.base_model)[0] in ("SegForestNet", "ForestEnsemble", "ResidualModel")
        self.full_gradient_flow = getattr(config, "full_gradient_flow", False)
        if self.full_gradient_flow:
            self.loss_weight = config.loss_weight
            assert 0 <= self.loss_weight and self.loss_weight <= 1
        
        self.base_model = core.create_object(
            models, config.base_model,
            input_shape = params.input_shape,
            num_classes = params.num_classes
        )
        if hasattr(config, "weights"):
            print(f"loading model weights from '{config.weights}' ...")
            with core.open(config.weights, "rb") as f:
                self.base_model.load_state_dict(torch.load(f, map_location=core.device))
        self.model_prepare_funcs = [self.base_model.prepare_for_epoch]
        
        true_base_model = config.base_model
        while core.get_object_meta_info(true_base_model)[0] != "SegForestNet":
            true_base_model = core.get_object_meta_info(true_base_model)[1].base_model
        self.res_model = core.create_object(
            models, true_base_model,
            input_shape = params.input_shape,
            num_classes = params.num_classes
        )
        del self.res_model.encoder
        del self.res_model.encoder_suffix
        self.model_prepare_funcs.append(self.res_model.prepare_for_epoch)
        self.model_prepare_funcs = tuple(self.model_prepare_funcs)

    def prepare_for_epoch(self, epoch, epochs):
        for func in self.model_prepare_funcs:
            func(epoch, epochs)
        
    def forward(self, x, yt=None, ce_loss_func=None, weight=None, ignore_index=None, z=None, return_features=False):
        if self.full_gradient_flow:
            return self.forward_with_full_gradient_flow(x, yt, ce_loss_func, weight, ignore_index, z, return_features)
        if z is None:
            with torch.no_grad():
                x, z = self.base_model(x, return_features=True)
                self.sample_points = self.base_model.sample_points
                self.res_model.sample_points = self.sample_points
        if return_features:
            return x, z
        with torch.no_grad():
            yp = self.base_model(x=x, z=z)
        x = [x.detach() for x in x]
        z = z.detach()
        yp = yp.detach()
        yp += self.res_model(x=x, z=z)
        if yt is None:
            return yp
        return yp, ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)

    def forward_with_full_gradient_flow(self, x, yt, ce_loss_func, weight, ignore_index, z, return_features):
        if z is None:
            x, z = self.base_model(x, return_features=True)
            self.sample_points = self.base_model.sample_points
            self.res_model.sample_points = self.sample_points
        if return_features:
            return x, z
        yp = self.base_model(
            x=x, yt=yt, ce_loss_func=ce_loss_func, weight=weight, ignore_index=ignore_index, z=z
        )
        if not (yt is None):
            yp, loss = yp
        yp += self.res_model(x=x, z=z)
        if yt is None:
            return yp
        loss = (1-self.loss_weight)*loss + self.loss_weight*ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
        return yp, loss

    def feature_partitioning_meta(self, partitioning_type):
        base_model = self.base_model
        while not isinstance(base_model, models.SegForestNet):
            base_model = base_model.base_model
        return base_model.feature_partitioning_meta(partitioning_type)
    
    def features_to_partitions(self, z, partitioning_type):
        base_model = self.base_model
        while not isinstance(base_model, models.SegForestNet):
            base_model = base_model.base_model
        return base_model.features_to_partitions(z, partitioning_type)
    
    def partitions_to_features(self, p, partitioning_type):
        base_model = self.base_model
        while not isinstance(base_model, models.SegForestNet):
            base_model = base_model.base_model
        return base_model.partitions_to_features(p, partitioning_type)
