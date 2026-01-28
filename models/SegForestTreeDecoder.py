import core
import numpy as np
import torch
import torch.nn as nn
import utils
import types


class TreeFeatureDecoder(nn.ModuleList):
    def __init__(self, config, encoder, num_input_features, num_output_features, effective_num_input_features = -1, is_shape_decoder = False):
        super().__init__()
        self.use_residuals = config.decoder.use_residual_blocks
        use_bias = getattr(config.decoder, "use_bias", encoder.use_bias)
        c = (config.decoder.context, 2*config.decoder.context + 1)
        f = config.decoder.intermediate_features
        assert type(c[0]) == int and 0 <= c[0]
        
        if effective_num_input_features < 0:
            effective_num_input_features = num_input_features
        
        vq_config = config.decoder.vq.__dict__.copy()
        vq_config["type"] = vq_config["type"][0 if is_shape_decoder else 1]
        if num_input_features == effective_num_input_features:
            vq_layer = utils.create_vector_quantization_layer(feature_size=num_input_features, **vq_config)
        else:
            assert num_input_features > effective_num_input_features
            
            class PartialVectorQuantization(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vq_layer = utils.create_vector_quantization_layer(feature_size=effective_num_input_features, **vq_config)
                    self.num_other_features = num_input_features - effective_num_input_features
                    
                def prepare_for_epoch(self, epoch, epochs):
                    self.vq_layer.prepare_for_epoch(epoch, epochs)
                    
                def forward(self, x):
                    x = x[:,:self.num_other_features], self.vq_layer(x[:,self.num_other_features:])
                    self.loss = self.vq_layer.loss
                    return torch.cat(x, dim=1)
            
            vq_layer = PartialVectorQuantization()
                
        self.append(
            nn.Sequential(
                vq_layer,
                nn.Conv2d(num_input_features, f, 1, bias=use_bias),
                encoder.normalization(f, affine=use_bias)
            ).to(core.device)
        )
        
        for i in range(config.decoder.num_blocks):
            block = [encoder.activation(f)]
            if c[0] > 0:
                block.extend([
                    encoder.padding(c[0]),
                    nn.Conv2d(f, f, c[1], groups=f, bias=use_bias),
                    encoder.normalization(f, affine=use_bias),
                    encoder.activation(f)
                ])
            block.extend([
                nn.Conv2d(f, f, 1, bias=use_bias),
                encoder.normalization(f, affine=use_bias)
            ])
            self.append(
                nn.Sequential(*block).to(core.device)
            )
        
        self.append(
            nn.Sequential(
                encoder.activation(f),
                nn.Conv2d(f, num_output_features, 1, bias=use_bias)
            ).to(core.device)
        )

        if encoder.self_normalizing:
            weights = self.state_dict()
            for k, w in weights.items():
                nn.init.normal_(w, mean=0, std=1/np.sqrt(np.product(w.shape)))
                weights[k] = w
            self.load_state_dict(weights)
            
    def forward(self, x):
        for i, layer in enumerate(self):
            if hasattr(self, "adaptation_hooks") and i > 0 and i < len(self)-1 and not (self.adaptation_hooks.funcs[i-1] is None):
                z = layer[:-1](x)
                label = f"{self.adaptation_hooks.prefix}{i-1}"
                if not (label in self.adaptation_hooks.data.parameters):
                    self.adaptation_hooks.data.parameters[label] = types.SimpleNamespace()
                z = self.adaptation_hooks.funcs[i-1](
                    z,
                    self.adaptation_hooks.data.parameters[label],
                    self.adaptation_hooks.data.config.in_decoders
                )
                z = layer[-1](z)
            else:
                x = layer(x) if (i==0 or i+1 == len(self) or not self.use_residuals) else layer(x) + x
        return x
    
    def set_adaptation_hooks(self, prefix, hooks, data_storage):
        assert len(self) == len(hooks) + 2
        self.adaptation_hooks = types.SimpleNamespace(prefix=prefix, funcs=hooks, data=data_storage)

def BSPSegNetDecoder(config, encoder, num_input_features, num_output_features, effective_num_input_features = -1, is_shape_decoder = False):
    use_bias = getattr(config.decoder, "use_bias", encoder.use_bias)
    features = [num_input_features]
    features.extend(config.decoder.intermediate_features)
    
    decoder = [nn.Sequential(utils.create_vector_quantization_layer(type=0))]
    for i in range(len(features)-1):
        decoder.extend([
            nn.Conv2d(features[i], features[i+1], 1, bias=use_bias),
            encoder.normalization(features[i+1], affine=use_bias),
            encoder.activation(features[i+1])
        ])
    decoder.append(nn.Conv2d(features[-1], num_output_features, 1, bias=use_bias))
    
    return nn.Sequential(*decoder).to(core.device)
