import core
import numpy as np
import torch
import torch.nn as nn
import types
import utils
import models
import functools
from .SegForestTree import PartitioningTree


class SegForestNet(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        if params.input_shape[1] != params.input_shape[2]:
            raise RuntimeError("input shape for SegForestNet must be square")
        self.softmax_temperature = utils.get_scheduler(config.region_map.softmax_temperature)
        
        if len(config.trees) == 1 and getattr(config.trees[0], "one_tree_per_class", False):
            assert not hasattr(config.trees[0], "outputs")
            tree = config.trees[0].__dict__
            tree["outputs"] = [0]
            for i in range(1, params.num_classes):
                config.trees.append(types.SimpleNamespace(**tree.copy()))
                config.trees[-1].outputs = [i]
            if "share_architecture" in tree:
                if tree["share_architecture"]:
                    for tree in config.trees[1:]:
                        tree.share_architecture = 0
                else:
                    for tree in config.trees[1:]:
                        del tree.share_architecture
                del config.trees[0].share_architecture
        
        class_to_tree_map = [-1] * params.num_classes
        num_encoder_output_features = 0
        for i, tree in enumerate(config.trees):
            if not hasattr(tree, "outputs"):
                assert len(config.trees) == 1
                tree.outputs = list(range(params.num_classes))
            if hasattr(tree, "share_architecture"):
                assert tree.share_architecture < i
                for k, v in config.trees[tree.share_architecture].__dict__.items():
                    if k == "outputs":
                        continue
                    tree.__dict__[k] = v
            if type(tree.outputs) != np.ndarray:
                for output in tree.outputs:
                    assert type(output) == int
                    assert 0 <= output < params.num_classes
                tree.outputs = np.asarray(tree.outputs, dtype=np.int32)
            for output in tree.outputs:
                assert class_to_tree_map[output] == -1
                class_to_tree_map[output] = i
            tree.actual_num_features = types.SimpleNamespace(**tree.num_features.__dict__)
            if config.decoder.vq.type[0] == 7: # references VQGumbel in create_vector_quantization_layer in utils/vectorquantization.py
                tree.num_features.shape = config.decoder.vq.codebook_size
            if config.decoder.vq.type[1] == 7: # references VQGumbel in create_vector_quantization_layer in utils/vectorquantization.py
                tree.num_features.content = config.decoder.vq.codebook_size
            num_encoder_output_features += tree.num_features.shape + tree.num_features.content
        assert -1 not in tuple(class_to_tree_map)

        self.encoder = models.feature_extractor(config.encoder, params.input_shape[0])
        self.encoder.downsampling_factor = 2**self.encoder.downsampling
        self.encoder.feature_size = params.input_shape[-1] // self.encoder.downsampling_factor
        encoder_suffix_input_features = self.encoder.num_features[-1]
        if config.features.dct_sidechannel.type > 0:
            dct_sidechannel = [
                nn.Conv2d(
                    params.input_shape[0] * self.encoder.downsampling_factor**2,
                    config.features.dct_sidechannel.num_features,
                    1, bias=self.encoder.use_bias
                ),
                self.encoder.normalization(config.features.dct_sidechannel.num_features, affine=self.encoder.use_bias)
            ]
            for _ in range(config.features.dct_sidechannel.num_res_blocks):
                dct_sidechannel.append(nn.Sequential(
                    self.encoder.activation(config.features.dct_sidechannel.num_features),
                    nn.Conv2d(config.features.dct_sidechannel.num_features, config.features.dct_sidechannel.num_features, 1, bias=self.encoder.use_bias),
                    self.encoder.normalization(config.features.dct_sidechannel.num_features, affine=self.encoder.use_bias),
                    self.encoder.activation(config.features.dct_sidechannel.num_features),
                    nn.Conv2d(config.features.dct_sidechannel.num_features, config.features.dct_sidechannel.num_features, 1, bias=self.encoder.use_bias),
                    self.encoder.normalization(config.features.dct_sidechannel.num_features, affine=self.encoder.use_bias)
                ))
            dct_sidechannel.append(self.encoder.activation(config.features.dct_sidechannel.num_features))
            self.dct_sidechannel_dropout = config.features.dct_sidechannel.dropout
            assert self.dct_sidechannel_dropout < 1
            self.dct_sidechannel = nn.Sequential(*dct_sidechannel).to(core.device)
            encoder_suffix_input_features += config.features.dct_sidechannel.num_features
            self.requires_dct_input = config.features.dct_sidechannel.type
        else:
            self.requires_dct_input = 0
        encoder_suffix = [self.encoder.padding(config.features.context)] if config.features.context > 0 else []
        encoder_suffix.extend([
            nn.Conv2d(encoder_suffix_input_features, num_encoder_output_features, 1+2*config.features.context, bias=self.encoder.use_bias),
            self.encoder.normalization(num_encoder_output_features, affine=self.encoder.use_bias),
            self.encoder.activation(num_encoder_output_features)
        ])
        self.encoder_suffix = nn.Sequential(*encoder_suffix).to(core.device)        
        self.model_name = "SegForestNet"
        
        if getattr(params, "encoder_only", False):
            return
        
        self.output_shape = (params.num_classes, *params.input_shape[1:])
        self.trees = tuple([PartitioningTree(config, params, self.encoder, tree) for tree in config.trees])
        
        self.num_tree_parameters = 0
        for i, tree in enumerate(self.trees):
            self.num_tree_parameters += tree.num_tree_parameters
            if hasattr(tree.config, "share_architecture"):
                other_tree = self.trees[tree.config.share_architecture]
                tree.decoder0 = other_tree.decoder0
                tree.decoder1 = other_tree.decoder1
                if hasattr(tree, "region_map_rendering_params"):
                    tree.region_map_rendering_params = other_tree.region_map_rendering_params
                    tree.region_map_rendering.node_weight[1] = other_tree.region_map_rendering_params
                if hasattr(tree, "classifier"):
                    tree.classifier = other_tree.classifier
            else:
                setattr(self, f"_PartitioningTree{i}_decoder0_", tree.decoder0)
                setattr(self, f"_PartitioningTree{i}_decoder1_", tree.decoder1)
                if hasattr(tree, "region_map_rendering_params"):
                    setattr(self, f"_PartitioningTree{i}_region_map_rendering_params_", tree.region_map_rendering_params)
                if hasattr(tree, "classifier"):
                    setattr(self, f"_PartitioningTree{i}_classifier_", tree.classifier)
                    
        self.weight_quantization = config.weight_quantization
        if self.weight_quantization.enabled:
            assert self.weight_quantization.bits > 1
            assert len(self.weight_quantization.loss_weights) == 3
            assert np.sum(self.weight_quantization.loss_weights) < 1
            assert self.weight_quantization.eval_mode in (0, 1, 2) # 0 - no quant.; 1 - quant. for eval but restore weights for the next epoch; 2 - quant. for eval and keep quantized weights for the next epoch
            self.weight_quantization.loss_func = getattr(nn.functional, self.weight_quantization.loss_func)
            self.weight_quantization.loss_weights = (
                np.sum(self.weight_quantization.loss_weights),
                *self.weight_quantization.loss_weights
            )
            self.weight_quantization.zero_threshold = 2**(self.weight_quantization.min_exp-0.5)
            self.weight_quantization.max_exp_threshold = self.weight_quantization.min_exp + 2**(self.weight_quantization.bits-1) - 1.5
            self.weight_quantization.is_quantized = False
        
        config = config.loss
        assert config.cross_entropy in ("pixels", "leaves", "leaves_argmax", "leaves_stop", "leaves_argmax_stop")
        assert len(config.weights) == 5
        if config.min_region_size <= 0:
            config.weights[2] = 0
                
        self.tree_weights = [tree.config.outputs.shape[0]/params.num_classes for tree in self.trees]
        self.loss_weights = [config.weights[i]/np.sum(config.weights) for i in range(len(config.weights))]
                
        if config.cross_entropy != "pixels":
            apply_argmax = "argmax" in config.cross_entropy
            apply_detach = "stop" in config.cross_entropy
            self.ce_loss_func = lambda yts, w, ic: self.leaf_loss(yts, apply_argmax, apply_detach, w, ic)
        self.dist_func = getattr(SegForestNet, f"dist_metric_{config.distribution_metric}")
        self.min_region_size = config.min_region_size
        
        if self.encoder.self_normalizing:
            weights = self.state_dict()
            for k, w in weights.items():
                if k[:8] == "encoder.":
                    continue
                nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
                weights[k] = w
            self.load_state_dict(weights)
                    
    def prepare_for_epoch(self, epoch, epochs):
        self.softmax_temperature(epoch, epochs)
        for i, tree in enumerate(self.trees):
            tree.region_map_rendering.softmax_temperature = self.softmax_temperature.value
            if hasattr(tree.config, "share_architecture"):
                continue
            for decoder in (tree.decoder0, tree.decoder1):
                decoder[0][0].prepare_for_epoch(epoch, epochs) # vector quantization layer
        
        if (not self.weight_quantization.enabled) or self.weight_quantization.eval_mode == 0:
            return
        
        if self.training:
            if hasattr(self, "unquantized_weights"):
                for k, v in self.unquantized_weights.items():
                    self.unquantized_weights[k] = v.to(core.device)
                self.load_state_dict(self.unquantized_weights)
                del self.unquantized_weights
            self.weight_quantization.is_quantized = False # quantization is not guaranteed after another epoch of training
        elif (not self.weight_quantization.is_quantized) and epoch >= self.weight_quantization.unquantized_eval_epochs:
            if self.weight_quantization.eval_mode == 1:
                self.unquantized_weights = self.state_dict().copy()
                for k, v in self.unquantized_weights.items():
                    self.unquantized_weights[k] = v.detach().clone().cpu()
            for w in self.parameters():
                w[w.abs() <= self.weight_quantization.zero_threshold] = 0
                for sign, i in (
                    (-1, w < -self.weight_quantization.zero_threshold),
                    (1, w > self.weight_quantization.zero_threshold)
                ):
                    log_w = (sign * w[i]).log2()
                    log_w[log_w >= self.weight_quantization.max_exp_threshold] = self.weight_quantization.max_exp_threshold - 0.5
                    w[i] = sign * torch.pow(2 * torch.ones_like(log_w), log_w.round())
            self.weight_quantization.is_quantized = True
    
    def forward(
        self, x,
        yt=None, ce_loss_func=None, weight=None, ignore_index=None,
        x_vis=None, lut=None, region_visualization_path="",
        z=None, return_features=False, x_dct=None,
        target_yp=None, target_yp_mask=None
    ):
        if z is None:
            if (not hasattr(self, "sample_points")) or self.sample_points.shape[0] < x.shape[0]:
                self.create_sample_points(x)

            x = [x, *self.encoder(x)]
            if self.requires_dct_input > 0:
                x_dct = self.dct_sidechannel[:2](x_dct.reshape(x_dct.shape[0], -1, *x_dct.shape[-2:]))
                for dct_model_layer in self.dct_sidechannel[2:-1]:
                    x_dct = x_dct + dct_model_layer(x_dct)
                x_dct = self.dct_sidechannel[-1](x_dct)
                if self.dct_sidechannel_dropout > 0:
                    x_dct = self.encoder.dropout.func(x_dct, self.dct_sidechannel_dropout, self.training)
                z = self.encoder_suffix(torch.cat((x[-1], x_dct), dim=1))
            else:
                z = self.encoder_suffix(x[-1])
            if hasattr(self, "adaptation_hook") and not (self.adaptation_hook.func is None):
                z = self.adaptation_hook.func(
                    z, self,
                    self.adaptation_hook.data.base_parameters,
                    self.adaptation_hook.data.config.pre_decoders
                )
            if return_features:
                return x, z
        
        for i in {tree.config.classifier_skip_from for tree in self.trees if hasattr(tree.config, "classifier_skip_from") and tree.config.classifier_skip_from > 0}:
            x[i] = nn.functional.interpolate(x[i], x[0].shape[2:], mode="bilinear")
        sample_points = self.sample_points[:x[0].shape[0]]

        yp = torch.zeros((x[0].shape[0], *self.output_shape), dtype=torch.float32, device=x[0].device)
        for i, tree in enumerate(self.trees):
            z = tree.render(x, z, yp, sample_points, self.training)
        if yt is None:
            return yp

        yt_onehot = nn.functional.one_hot(yt, num_classes=max(self.output_shape[0], ignore_index+1))
        region_dists = [tree.get_region_distributions(yt_onehot, ignore_class=ignore_index) for tree in self.trees]
        if target_yp is None:
            if hasattr(self, "ce_loss_func"):
                ce_loss = self.ce_loss_func([p for p,_ in region_dists], weight, ignore_index)
            else:
                ce_loss = ce_loss_func(yp, yt, weight=weight, ignore_index=ignore_index)
        else:
            i = target_yp_mask.nonzero().t()
            ce_loss = ce_loss_func(yp[i[0], :, i[1], i[2]], target_yp[i[0], :, i[1], i[2]])
                
        if len(region_visualization_path) > 0:
            self.visualize_regions(x_vis, yp, yt, lut, region_dists, region_visualization_path)
            
        loss = [(
            w * self.dist_func(p).mean(),
            w * torch.maximum(self.min_region_size-s, torch.zeros(s.shape,device=s.device)).mean(),
            w * self.dist_func(tree.region_map.permute(0,2,3,1).clone()).mean(),
            #w * tree.cov.abs().mean(),
            w * functools.reduce(lambda x,y: x+y, (*tree.decoder0[0][0].loss, *tree.decoder1[0][0].loss))
        ) for i,(w,tree,(p,s)) in enumerate(zip(self.tree_weights,self.trees,region_dists))]
        loss = functools.reduce(lambda xs,ys: tuple([x+y for x,y in zip(xs,ys)]), loss)
            
        # loss[0] -> region distribution
        # loss[1] -> minimum region size
        # loss[2] -> region map distribution
        # loss[3] -> vector quantization
            
        loss = self.loss_weights[0]*ce_loss + \
            self.loss_weights[1]*loss[0] + \
            self.loss_weights[2]*loss[1] + \
            self.loss_weights[3]*loss[2] + \
            self.loss_weights[4]*loss[3]
        
        if self.weight_quantization.enabled:
            loss = loss, [], []
            loss_func = self.weight_quantization.loss_func
            for w in self.parameters():
                loss[1].append(w[w.abs() <= self.weight_quantization.zero_threshold].abs())
                loss[2].append(-w[w < -self.weight_quantization.zero_threshold])
                loss[2].append(w[w > self.weight_quantization.zero_threshold])
            loss = loss[0], torch.cat(loss[1]), torch.cat(loss[2])
            if loss[1].shape[0] > 0:
                loss = loss[0], loss_func(loss[1], torch.zeros_like(loss[1], requires_grad=False)), loss[2]
            else:
                loss = loss[0], 0, loss[2]
            if loss[2].shape[0] > 0:
                log_w = loss[2].log2()
                log_w_overflow = log_w[log_w >= self.weight_quantization.max_exp_threshold]
                low_w = log_w[log_w < self.weight_quantization.max_exp_threshold]
                loss = *loss[:2], loss_func(log_w, log_w.detach().round())
                loss = *loss, loss_func(log_w_overflow - self.weight_quantization.max_exp_threshold, torch.zeros_like(log_w_overflow, requires_grad=False))
            else:
                loss = *loss[:2], 0, 0
            loss = [weight*loss for weight, loss in zip(self.weight_quantization.loss_weights, loss)]
            loss = loss[0] + loss[1] + loss[2] + loss[3]
        
        return yp, loss
    
    def create_sample_points(self, x):
        n = self.encoder.downsampling_factor
        self.sample_points = np.asarray(np.arange(n), dtype=np.float32)
        self.sample_points = (2*self.sample_points + 1) / (2*n)
        self.sample_points = np.meshgrid(self.sample_points, self.sample_points)
        self.sample_points = [np.expand_dims(p,axis=0) for p in self.sample_points]
        self.sample_points = np.concatenate(self.sample_points, axis=0)
        self.sample_points = np.expand_dims(self.sample_points, axis=0)
        self.sample_points = np.tile(self.sample_points, [x.shape[0], 1, self.encoder.feature_size, self.encoder.feature_size])
        self.sample_points = torch.from_numpy(self.sample_points).float().to(x.device)
        # shape: [batch size, x_or_y, height, width]
        # x_or_y == 0 -> x coordinates
        # x_or_y == 1 -> y coordinates

    def leaf_loss(self, yts, apply_argmax, apply_detach, class_weights, ignore_class):
        loss = []
        for tree_weight, tree, yt in zip(self.tree_weights, self.trees, yts):
            yp = tree.tree_parameters[1]
            yp = yp.reshape(yp.shape[0]*yt.shape[0], yp.shape[1]//yt.shape[0], *yp.shape[2:4])
            if tree.config.outputs.shape[0] != self.output_shape[0]:
                temp = torch.empty([yp.shape[0], yp.shape[1]+1, *yp.shape[2:]], dtype=yp.dtype, device=core.device)
                temp[:,:-1] = yp
                temp[:,-1] = config.ce_constant
                yp = temp
            yp = nn.functional.log_softmax(yp, dim=1)
                
            yt = yt.permute(1, 0, 4, 2, 3)
            yt = yt.reshape(yt.shape[0]*yt.shape[1], *yt.shape[2:])
            if apply_argmax:
                yt = nn.functional.one_hot(yt.argmax(1), num_classes=yt.shape[1]).permute(0, 3, 1, 2)
            if apply_detach:
                yt = yt.detach()
                    
            ce = -yt * yp
            for i, class_weight in enumerate(class_weights[tree.config.outputs]):
                ce[:,i] *= class_weight
            if ignore_class < ce.shape[1]:
                ce[:,ignore_class] *= 0
                    
            loss.append(tree_weight * ce.sum(1).mean())
        return functools.reduce(lambda x,y: x+y, loss)
            
    @staticmethod
    def dist_metric_entropy(p):
        p[p<10**-6] = 10**-6
        return -(p * p.log()).sum(-1)
                
    @staticmethod
    def dist_metric_gini(p):
        return 1 - (p**2).sum(-1)
    
    def visualize_regions(self, x_vis, yp, yt, lut, region_dists, path):
        with torch.no_grad():
            yp = yp.detach().argmax(1).cpu().numpy()
            yt = yt.cpu().numpy()
            images = [(self.tree_weights[i] * self.dist_func(p)).detach() for i,(p,s) in enumerate(region_dists)]
            images = [img.repeat_interleave(self.trees[i].downsampling_factor,3) for i, img in enumerate(images)]
            images = [img.repeat_interleave(self.trees[i].downsampling_factor,2) for i, img in enumerate(images)]
            region_maps = [tree.region_map.clone().detach().permute(1, 0, 2, 3) for tree in self.trees]
            images = [img*region_map for img, region_map in zip(images, region_maps)]
            for img in images:
                for i in range(1, img.shape[0]):
                    img[0] += img[i]
            region_loss_images = [img[0].cpu().numpy() for img in images]
            
            region_images = []
            for _, region_map in enumerate(region_maps):
                region_map = region_map.cpu().numpy()
                regions = [utils.hsv2rgb(i/region_map.shape[0],1,1) for i in range(region_map.shape[0])]
                regions = [np.expand_dims(np.asarray(region), (1,2)) for region in regions]
                regions = [[region*region_map[i,j] for i, region in enumerate(regions)] for j in range(region_map.shape[1])]
                regions = [functools.reduce(lambda x,y: x+y, r) for r in regions]
                regions = np.asarray(regions, dtype=np.uint8)
                region_images.append(np.moveaxis(regions, 1, 3))
        
        import matplotlib.pyplot as plt
        base_size = 4
        n = 3 + 2*len(self.trees)
        fig, axes = plt.subplots(x_vis.shape[0], n, figsize=(n*base_size, x_vis.shape[0]*base_size))
        for i in range(x_vis.shape[0]):
            axes[i,0].imshow(np.moveaxis(x_vis[i], 0, 2))
            axes[i,1].imshow(lut[yt[i]])
            axes[i,2].imshow(lut[yp[i]])
            for j in range(len(self.trees)):
                fig.colorbar(axes[i,3+2*j].imshow(region_loss_images[j][i]), ax=axes[i,3+2*j])
                axes[i,4+2*j].imshow(region_images[j][i])
            if i == 0:
                for j in range(n):
                    if j < 3:
                        axes[i,j].set_title(("input", "ground truth", "prediction")[j])
                    else:
                        k = j - 3
                        k = k//2, ("region loss", "regions")[k%2]
                        axes[i,j].set_title(f"tree {k[0]}: {k[1]}")
            for j in range(n):
                axes[i,j].set_xticks(())
                axes[i,j].set_yticks(())
        fig.tight_layout()
        fig.savefig(f"{path}.pdf")
        plt.close(fig)
        
    def feature_partitioning_meta(self, partitioning_type):
        partitioning_type = ("single", "trees", "semantics", "both").index(partitioning_type)
        if partitioning_type == 0:
            layout = (self.encoder_suffix[-3].out_channels,)
        elif partitioning_type == 1:
            layout = tuple([
                tree.config.num_features.shape + tree.config.num_features.content for tree in self.trees
            ])
        elif partitioning_type == 2:
            layout = (
                sum([tree.config.num_features.shape for tree in self.trees]),
                sum([tree.config.num_features.content for tree in self.trees])
            )
        else:
            layout = [tree.config.num_features.shape for tree in self.trees]
            layout.extend([tree.config.num_features.content for tree in self.trees])
            layout = tuple(layout)
        return partitioning_type, layout
    
    def features_to_partitions(self, z, partitioning_type):
        if partitioning_type == 0:
            return (z,)
        shape, content = [], []
        for tree in self.trees:
            shape.append(z[:,:tree.config.num_features.shape])
            content.append(z[:,-tree.config.num_features.content:])
            z = z[:,tree.config.num_features.shape:-tree.config.num_features.content]
        if partitioning_type == 1:
            z = tuple([torch.cat((s, c), dim=1) for s, c in zip(shape, content)])
        elif partitioning_type == 2:
            z = torch.cat(shape, dim=1), torch.cat(content, dim=1)
        else:
            shape.extend(content)
            z = tuple(shape)
        return z
    
    def partitions_to_features(self, p, partitioning_type):
        if partitioning_type == 0:
            return p[0]
        elif partitioning_type == 1:
            shape, content = [], []
            for pi, tree in zip(p, self.trees):
                shape.append(pi[:,:tree.config.num_features.shape])
                content.append(pi[:,tree.config.num_features.shape:])
        elif partitioning_type == 2:
            shape, content = [], []
            for tree in self.trees:
                n = tree.config.num_features.shape, tree.config.num_features.content
                shape.append(p[0][:,:n[0]])
                content.append(p[1][:,:n[1]])
                p = p[0][:,n[0]:], p[1][:,n[1]:]
        else:
            n = len(p) // 2
            shape, content = p[:n], list(p[n:])
        content.reverse()
        return torch.cat((*shape, *content), dim=1)
    
    def set_adaptation_hooks(self, prefix, base_hook, shape_hooks, content_hooks, tree_hooks, data_storage):
        self.adaptation_hook = types.SimpleNamespace(func=base_hook, data=data_storage)
        for i, tree in enumerate(self.trees):
            tree.set_adaptation_hooks(f"{prefix}tree{i}_", shape_hooks, content_hooks, tree_hooks, data_storage)
