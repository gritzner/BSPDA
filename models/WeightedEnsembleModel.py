import core
import torch
import torch.nn as nn
import models


class WeightedEnsembleModel(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        assert core.get_object_meta_info(config.base_model)[0] in ("ForestEnsemble", "ResidualModel")
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
        
        self.the_models = [self.base_model]
        while isinstance(self.the_models[-1], models.ResidualModel):
            self.the_models.append(self.the_models[-1].base_model)
        assert isinstance(self.the_models[-1], models.ForestEnsemble)
        self.the_models.reverse()
        
        n = self.the_models[0].num_extra_branches + len(self.the_models)
        self.ensemble_weights = nn.Parameter(torch.ones((n, 1, 1, 1, 1), dtype=torch.float32, device=core.device))
        if config.affine:
            self.ensemble_bias = nn.Parameter(torch.zeros((n, 1, 1, 1, 1), dtype=torch.float32, device=core.device))
        
    def prepare_for_epoch(self, epoch, epochs):
        self.base_model.prepare_for_epoch(epoch, epochs)
        
    def forward(self, x, yt=None, ce_loss_func=None, weight=None, ignore_index=None):
        if self.full_gradient_flow:
            return self.forward_with_full_gradient_flow(x, yt, ce_loss_func, weight, ignore_index)
        with torch.no_grad():
            x, z = self.the_models[0](x, return_features=True)
            for model in self.the_models[1:]:
                model.sample_points = self.the_models[0].sample_points
                model.res_model.sample_points = model.sample_points
        x = [x.detach() for x in x]
        z = z.detach()
        yp = self.the_models[0](x=x, z=z, stack=True).detach()
        yp = yp * self.ensemble_weights[:yp.shape[0]]
        if hasattr(self, "ensemble_bias"):
            yp = yp + self.ensemble_bias[:yp.shape[0]]
        yp = yp.mean(0)
        if len(self.the_models) > 1:
            res_yp = torch.stack([model.res_model(x=x, z=z) for model in self.the_models[1:]]).detach()
            res_yp = res_yp * self.ensemble_weights[-res_yp.shape[0]:]
            if hasattr(self, "ensemble_bias"):
                res_yp = res_yp + self.ensemble_bias[-res_yp.shape[0]:]
            yp = yp + res_yp.sum(0)
        if yt is None:
            return yp
        return yp, ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)

    def forward_with_full_gradient_flow(self, x, yt, ce_loss_func, weight, ignore_index):
        x, z = self.the_models[0](x, return_features=True)
        for model in self.the_models[1:]:
            model.sample_points = self.the_models[0].sample_points
            model.res_model.sample_points = model.sample_points
        yp = self.the_models[0](
            x=x, yt=yt, ce_loss_func=ce_loss_func, weight=weight, ignore_index=ignore_index, z=z, stack=True
        )
        if not (yt is None):
            yp, loss = yp
        yp = yp * self.ensemble_weights[:yp.shape[0]]
        if hasattr(self, "ensemble_bias"):
            yp = yp + self.ensemble_bias[:yp.shape[0]]
        yp = yp.mean(0)
        if len(self.the_models) > 1:
            res_yp = [model.res_model(
                x=x, yt=yt, ce_loss_func=ce_loss_func, weight=weight, ignore_index=ignore_index, z=z
            ) for model in self.the_models[1:]]
            if not (yt is None):
                for lossi in [res_yp[1] for res_yp in res_yp]:
                    loss = loss + lossi
                res_yp = [res_yp[0] for res_yp in res_yp]
            res_yp = torch.stack(res_yp)
            res_yp = res_yp * self.ensemble_weights[-res_yp.shape[0]:]
            if hasattr(self, "ensemble_bias"):
                res_yp = res_yp + self.ensemble_bias[-res_yp.shape[0]:]
            yp = yp + res_yp.sum(0)
        if yt is None:
            return yp
        loss = (1-self.loss_weight)*loss + self.loss_weight*ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
        return yp, loss
