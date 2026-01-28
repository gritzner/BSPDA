import core
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu, leaky_relu
import types
import functools
from .SegForestComponents import *
from .SegForestTreeDecoder import TreeFeatureDecoder, BSPSegNetDecoder


class OneHotArgMax(torch.autograd.Function):
    @staticmethod
    def forward(context, x):
        y = nn.functional.one_hot(x.argmax(1), num_classes=x.shape[1])
        return torch.permute(y.type(x.dtype), (0, 3, 1, 2))

    @staticmethod
    def backward(context, grad):
        return grad


class PartitioningTree():
    def __init__(self, config, params, encoder, tree):
        self.config = tree
        self.config.apply_argmax = getattr(self.config, "apply_argmax", False)
        self.downsampling_factor = encoder.downsampling_factor
        self.dropout_func = encoder.dropout.func
        self.region_map_rendering = types.SimpleNamespace(
            func = getattr(PartitioningTree, f"render_region_map_{config.region_map.accumulation}"),
            node_weight = config.region_map.node_weight
        )
        if config.region_map.accumulation == "mul2":
            self.region_map_rendering.node_weight[0] = getattr(
                PartitioningTree,
                f"distance_transform_{self.region_map_rendering.node_weight[0]}"
            )
            if self.region_map_rendering.node_weight[3]:
                self.region_map_rendering_params = nn.Parameter(torch.as_tensor(
                    self.region_map_rendering.node_weight[1],
                    dtype=torch.float32, device=core.device
                ))
                self.region_map_rendering.node_weight[1] = self.region_map_rendering_params
        self.per_region_outputs = tree.outputs.shape[0] if len(tree.classifier) == 0 else tree.classifier[0]
        
        self.inner_nodes = []
        num_params = 0
        self.num_leaf_nodes = 0

        queue = [eval(tree.graph)]
        while len(queue) > 0:
            node = queue.pop(0)
            assert config.region_map.accumulation != "mul2" or type(node) == BSPNode
            self.inner_nodes.append(node)
            num_params += node.num_params
            for i, child in enumerate(node.children):
                if child == Leaf:
                    node.children[i] = Leaf(self.num_leaf_nodes)
                    self.num_leaf_nodes += 1
                else:
                    queue.append(child)
        self.region_map_rendering.base_shape = (self.num_leaf_nodes, *params.input_shape[1:])
        
        num_params = max(num_params, 1)
        self.num_tree_parameters = num_params + self.num_leaf_nodes * tree.outputs.shape[0]

        for node in reversed(self.inner_nodes):
            node.indices = []
            for child in node.children:
                if type(child) == Leaf:
                    node.indices.append(np.asarray([child.index], dtype=np.int32))
                else:
                    node.indices.append(np.concatenate(child.indices))
            node.children = tuple(node.children)
            node.indices = tuple(node.indices)
        self.inner_nodes = tuple(self.inner_nodes)
        
        decoder_factory = globals()[getattr(config.decoder, "type", "TreeFeatureDecoder")]
        self.decoder0 = decoder_factory(
            config, encoder, tree.actual_num_features.shape, num_params, is_shape_decoder = True
        )
        assert tree.shape_to_content in (0, 1, 2)
        num_params = (0, tree.actual_num_features.shape, num_params)[tree.shape_to_content]
        self.decoder1 = decoder_factory(
            config, encoder, tree.actual_num_features.content + num_params,
            self.num_leaf_nodes * self.per_region_outputs, tree.actual_num_features.content
        )
        
        if len(tree.classifier) > 0:
            num_features = tree.classifier.copy()
            use_bias = getattr(tree, "classifier_use_bias", encoder.use_bias)
            if tree.classifier_skip_from == 0:
                num_features[0] += params.input_shape[0]
            elif tree.classifier_skip_from > 0:
                num_features[0] += encoder.num_features[tree.classifier_skip_from-1]
            self.classifier = []
            for in_features, out_features in zip(num_features[:-1], num_features[1:]):
                if tree.classifier_context > 0:
                    self.classifier.append(encoder.padding(tree.classifier_context))
                self.classifier.extend([
                    nn.Conv2d(in_features, out_features, 1+2*tree.classifier_context, bias=use_bias),
                    encoder.normalization(out_features, affine=use_bias),
                    encoder.activation(out_features)
                ])
            if tree.classifier_context > 0:
                self.classifier.append(encoder.padding(tree.classifier_context))
            self.classifier.append(nn.Conv2d(num_features[-1], tree.outputs.shape[0], 1+2*tree.classifier_context, bias=use_bias))
            self.classifier = nn.Sequential(*self.classifier).to(core.device)

    def render(self, x, z, y, sample_points, training):
        # seperate features into shape and content and decode them seperately
        shape, content, z = (
            z[:,:self.config.num_features.shape],
            z[:,-self.config.num_features.content:],
            z[:,self.config.num_features.shape:-self.config.num_features.content]
        )

        shape = self.dropout_func(shape, self.config.dropout.shape, training)
        content = self.dropout_func(content, self.config.dropout.content, training)
        
        if self.config.shape_to_content > 0:
            self.tree_parameters = (self.decoder0(shape), None) # delay decoding
            if self.config.shape_to_content == 1:
                self.tree_parameters = (
                    self.tree_parameters[0],
                    self.decoder1(torch.cat((self.decoder0[0][0](shape).detach(), content), dim=1))
                )
            elif self.config.shape_to_content == 2:
                self.tree_parameters = (
                    self.tree_parameters[0],
                    self.decoder1(torch.cat((self.tree_parameters[0].detach(), content), dim=1))
                )
        else:
            self.tree_parameters = (self.decoder0(shape), self.decoder1(content))

        #d_shape = shape - shape.mean(dim=(0, 2, 3))[None,:,None,None]
        #d_content = content - content.mean(dim=(0, 2, 3))[None,:,None,None]
        #self.cov = d_shape[:,:,None,:,:] * d_content[:,None,:,:,:]
        #self.cov = self.cov.mean(dim=(0, 3, 4))

        if hasattr(self, "adaptation_hooks"):
            shape, content = self.tree_parameters
            if not (self.adaptation_hooks.funcs[0] is None):
                label = f"{self.adaptation_hooks.prefix}shape"
                if not (label in self.adaptation_hooks.data.parameters):
                    self.adaptation_hooks.data.parameters[label] = types.SimpleNamespace()
                shape = self.adaptation_hooks.funcs[0](
                    shape,
                    self.adaptation_hooks.data.parameters[label],
                    self.adaptation_hooks.data.config.trees.shape
                )
            if not (self.adaptation_hooks.funcs[1] is None):
                label = f"{self.adaptation_hooks.prefix}content"
                if not (label in self.adaptation_hooks.data.parameters):
                    self.adaptation_hooks.data.parameters[label] = types.SimpleNamespace()
                content = self.adaptation_hooks.funcs[1](
                    content,
                    self.adaptation_hooks.data.parameters[label],
                    self.adaptation_hooks.data.config.trees.content
                )
            self.tree_parameters = shape, content
        
        # process shape features
        shape = self.tree_parameters[0].repeat_interleave(self.downsampling_factor, 3)
        shape = shape.repeat_interleave(self.downsampling_factor, 2)

        # render partitioning tree
        self.region_map = self.region_map_rendering.func(
            (x[0].shape[0], *self.region_map_rendering.base_shape), self.region_map_rendering.node_weight,
            self.region_map_rendering.softmax_temperature, shape, self.inner_nodes, sample_points,
        )
        if self.config.apply_argmax:
            self.region_map_soft = self.region_map
            self.region_map = OneHotArgMax.apply(self.region_map)
        
        # process content features
        content = self.tree_parameters[1].reshape(z.shape[0], self.num_leaf_nodes, self.per_region_outputs, *z.shape[2:])
        content = content.repeat_interleave(self.downsampling_factor, 4).repeat_interleave(self.downsampling_factor, 3)
        for i in range(self.num_leaf_nodes): # update prediction
            if hasattr(self, "classifier"):
                if self.config.classifier_skip_from >= 0:
                    logits = self.classifier(torch.cat((x[self.config.classifier_skip_from], content[:,i]), dim=1))
                else:
                    logits = self.classifier(content[:,i])
            else:
                logits = content[:,i]
            y[:,self.config.outputs] += self.region_map[:,i].unsqueeze(1) * logits
        
        return z
    
    @staticmethod
    def render_region_map_add(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.zeros(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_add(region_map, node_params, sample_points, node_weight)
        region_map = region_map / softmax_temperature
        return region_map.softmax(1)
    
    @staticmethod
    def render_region_map_mul(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.ones(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_mul(region_map, node_params, sample_points, node_weight)
        return region_map / region_map.sum(1, keepdim=True) 
    
    @staticmethod
    def render_region_map_mul2(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.empty(region_map_shape, dtype=torch.float32, device=sample_points.device)
        distance_maps = torch.empty((region_map_shape[0], 2, len(nodes), *region_map_shape[2:]), dtype=torch.float32, device=sample_points.device)
        
        transform_func, transform_weights, leaky_slope = node_weight[:3]
        for i, node in enumerate(nodes):
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            distance_maps[:,0,i] = node.sdf.compute(node_params, sample_points)
        distance_maps[:,1] = -distance_maps[:,0]
        distance_maps = transform_func(
            transform_weights[0] * distance_maps + transform_weights[1],
            transform_weights[2], leaky_slope
        )
        
        for i in range(region_map_shape[1]):
            distances = [distance_maps[:,0,j] for j,node in enumerate(nodes) if i in node.indices[0]]
            distances.extend([distance_maps[:,1,j] for j,node in enumerate(nodes) if i in node.indices[1]])
            region_map[:,i] = functools.reduce(lambda x,y: x*y, distances)
        
        region_map = region_map / softmax_temperature
        return region_map.softmax(1)
    
    @staticmethod
    def distance_transform_relu(distance_maps, weight, leaky_slope):
        return weight * relu(distance_maps)
    
    @staticmethod
    def distance_transform_leaky_relu(distance_maps, weight, leaky_slope):
        return weight * leaky_relu(distance_maps, negative_slope=leaky_slope)
    
    @staticmethod
    def distance_transform_sigmoid(distance_maps, weight, leaky_slope):
        return weight * torch.sigmoid(distance_maps)
    
    @staticmethod
    def distance_transform_clamp(distance_maps, weight, leaky_slope):
        return weight * torch.clamp(distance_maps, 0, 1)
    
    @staticmethod
    def distance_transform_leaky_clamp(distance_maps, weight, leaky_slope):
        distance_maps[distance_maps<0] *= leaky_slope
        i = distance_maps>1
        distance_maps[i] = 1 + (distance_maps[i] - 1) * leaky_slope
        return weight * distance_maps
    
    @staticmethod
    def distance_transform_smoothstep(distance_maps, weight, leaky_slope):
        distance_maps = torch.clamp(distance_maps, 0, 1)
        distance_maps = 3 * distance_maps**2 - 2 * distance_maps**3
        return weight * distance_maps
    
    @staticmethod
    def distance_transform_leaky_smoothstep(distance_maps, weight, leaky_slope):
        distance_maps[distance_maps<0] *= leaky_slope
        i = distance_maps>1
        distance_maps[i] = 1 + (distance_maps[i] - 1) * leaky_slope
        distance_maps = 3 * distance_maps**2 - 2 * distance_maps**3
        return weight * distance_maps
    
    @staticmethod
    def render_region_map_old(region_map_shape, node_weight, softmax_temperature, shape, nodes, sample_points):
        region_map = torch.ones(region_map_shape, dtype=torch.float32, device=sample_points.device)
        for node in nodes:
            node_params, shape = shape[:,:node.num_params], shape[:,node.num_params:]
            node.apply_old(region_map, node_params, sample_points, node_weight)
        region_map = region_map / softmax_temperature
        return region_map.softmax(1)
    
    def get_region_distributions(self, yt, ignore_class):
        downsampled_shape = (
            yt.shape[0],
            yt.shape[1] // self.downsampling_factor, self.downsampling_factor,
            yt.shape[2] // self.downsampling_factor, self.downsampling_factor,
            yt.shape[3]
        )
        
        ijk = torch.meshgrid((
            torch.arange(downsampled_shape[0], device=core.device),
            torch.arange(downsampled_shape[1], device=core.device),
            torch.arange(downsampled_shape[3], device=core.device),
        ), indexing="ij")
        ijk = [ijk.flatten() for ijk in ijk]
        
        p, s = [], []
        for i in range(self.num_leaf_nodes):
            pi = (yt * self.region_map[:, i, :, :, None]).reshape(*downsampled_shape).sum(-2).sum(-3)
            
            ignored = pi[ijk[0], ijk[1], ijk[2], ignore_class].clone()
            pi[ijk[0], ijk[1], ijk[2], ignore_class] = -1
            j = pi[ijk[0], ijk[1], ijk[2]].argmax(1)
            pi[ijk[0], ijk[1], ijk[2], j] += ignored
            pi[ijk[0], ijk[1], ijk[2], ignore_class] = 0
            
            si = pi.sum(-1)
            si[si < 10**-6] = 10**-6
            s.append(si)
            
            pi = pi / si.unsqueeze(-1)
            pi = pi[:, :, :, self.config.outputs]
            if pi.shape[-1] != yt.shape[-1]:
                # add artifical 'other' class, i.e., class that represents all classes not predicted by this tree
                temp = torch.empty([*pi.shape[:-1], pi.shape[-1]+1], dtype=pi.dtype, device=pi.device)
                temp[:, :, :, :-1] = pi
                temp[:, :, :, -1] = 1 - pi.sum(-1)
                pi = temp
            p.append(pi)
            
        return torch.stack(p), torch.stack(s)
    
    def set_adaptation_hooks(self, prefix, shape_hooks, content_hooks, tree_hooks, data_storage):
        self.decoder0.set_adaptation_hooks(f"{prefix}shape_decoder_layer", shape_hooks, data_storage)
        self.decoder1.set_adaptation_hooks(f"{prefix}content_decoder_layer", content_hooks, data_storage)
        self.adaptation_hooks = types.SimpleNamespace(prefix=prefix, funcs=tree_hooks, data=data_storage)
