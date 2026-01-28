import core
import numpy as np
import torch
import torch.nn as nn
import models


class ForestEnsemble(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        assert core.get_object_meta_info(config.base_model)[0] == "SegForestNet"
        assert config.size > 1
        self.num_extra_branches = config.size - 1
        assert 0 <= config.num_ignored_branches_during_training
        assert config.num_ignored_branches_during_training < self.num_extra_branches
        self.num_branches_during_training = self.num_extra_branches - config.num_ignored_branches_during_training
        self.rng = np.random.RandomState(core.random_seeds[config.rng_seed])
        
        self.base_model = core.create_object(
            models, config.base_model,
            input_shape = params.input_shape,
            num_classes = params.num_classes
        )
        self.model_prepare_funcs = [self.base_model.prepare_for_epoch]
        
        for i in range(self.num_extra_branches):
            extra_branch = core.create_object(
                models, config.base_model,
                input_shape = params.input_shape,
                num_classes = params.num_classes
            )
            del extra_branch.encoder
            del extra_branch.encoder_suffix
            setattr(self, f"_ExtraBranch{i}_", extra_branch)
            self.model_prepare_funcs.append(extra_branch.prepare_for_epoch)
        self.model_prepare_funcs = tuple(self.model_prepare_funcs)

    def prepare_for_epoch(self, epoch, epochs):
        for func in self.model_prepare_funcs:
            func(epoch, epochs)
        
    def forward(self, x, yt=None, ce_loss_func=None, weight=None, ignore_index=None, z=None, return_features=False, stack=False):
        if z is None:
            x, z = self.base_model(x, return_features=True)
            self.sample_points = self.base_model.sample_points
        
        if return_features:
            return x, z
        
        if yt is None:
            yp = [self.base_model(x=x, z=z)]
            for i in range(self.num_extra_branches):
                extra_branch = getattr(self, f"_ExtraBranch{i}_")
                extra_branch.sample_points = self.sample_points
                yp.append(extra_branch(x=x, z=z))
            if stack:
                return torch.stack(yp)
            for ypi in yp[1:]:
                yp[0] += ypi
            return yp[0] / (self.num_extra_branches+1)
        
        yp, loss = self.base_model(x=x, z=z, yt=yt, ce_loss_func=ce_loss_func, weight=weight, ignore_index=ignore_index)
        branches = self.rng.permutation(self.num_extra_branches)
        if self.training:
            branches = branches[:self.num_branches_during_training]
        if stack:
            yp = [yp]
        for i in branches:
            extra_branch = getattr(self, f"_ExtraBranch{i}_")
            extra_branch.sample_points = self.base_model.sample_points
            ypi, lossi = extra_branch(x=x, z=z, yt=yt, ce_loss_func=ce_loss_func, weight=weight, ignore_index=ignore_index)
            if stack:
                yp.append(ypi)
            else:
                yp += ypi
            loss += lossi
        if stack:
            return torch.stack(yp), loss
        return yp / (branches.shape[0]+1), loss

    def feature_partitioning_meta(self, partitioning_type):
        return self.base_model.feature_partitioning_meta(partitioning_type)
    
    def features_to_partitions(self, z, partitioning_type):
        return self.base_model.features_to_partitions(z, partitioning_type)
    
    def partitions_to_features(self, p, partitioning_type):
        return self.base_model.partitions_to_features(p, partitioning_type)

    def set_adaptation_hooks(self, prefix, base_hook, shape_hooks, content_hooks, tree_hooks, data_storage):
        self.base_model.set_adaptation_hooks(prefix, base_hook, shape_hooks, content_hooks, tree_hooks, data_storage)
        for i in range(self.num_extra_branches):
            extra_branch = getattr(self, f"_ExtraBranch{i}_")
            extra_branch.set_adaptation_hooks(f"{prefix}extra_branch{i}_", None, shape_hooks, content_hooks, tree_hooks, data_storage)
