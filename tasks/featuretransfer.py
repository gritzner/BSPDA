import core
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import utils
import time
import types
import cv2 as cv
import rust
import datetime


class F:
    @staticmethod
    def base_mean(z, model, parameters, config):
        z = model.features_to_partitions(z, config.partitioning)
        if not hasattr(parameters, "covs"):
            parameters.covs = [utils.CovarianceMatrix(zi.shape[1], False) for zi in z]
        for zi, cov in zip(z, parameters.covs):
            cov.add_to_mean(zi.sum(3).sum(2).sum(0))
            cov.num_samples += (zi.shape[0] * zi.shape[2] * zi.shape[3]) - 1
        return model.partitions_to_features(z, config.partitioning)
    
    @staticmethod
    def base_cov(z, model, parameters, config):
        for zi, cov in zip(model.features_to_partitions(z, config.partitioning), parameters.covs):
            zi = zi - cov.mean[None, :, None, None]
            for zi in zi:
                zi = zi[:, None] @ zi[None, :]
                cov.cov += zi.sum(3).sum(2)
        return z
    
    @staticmethod
    def base_adapt(z, model, parameters, config):
        z = model.features_to_partitions(z, config.partitioning)
        zs = []
        for zi, source, W, target in zip(z, parameters.source, parameters.Ws, parameters.target):
            zi = (zi - target.mean[None, :, None, None]).permute(1, 0, 2, 3)
            zi = (W @ zi.reshape(zi.shape[0], -1)).reshape(zi.shape).permute(1, 0, 2, 3)
            zs.append(zi + source.mean[None, :, None, None])
        return model.partitions_to_features(zs, config.partitioning)
    
    @staticmethod
    def layer_mean(z, parameters, config):
        if not hasattr(parameters, "cov"):
            parameters.cov = utils.CovarianceMatrix(z.shape[1], False)
            parameters.method = config.method
        parameters.cov.add_to_mean(z.sum(3).sum(2).sum(0))
        parameters.cov.num_samples += (z.shape[0] * z.shape[2] * z.shape[3]) - 1
        return z

    @staticmethod
    def layer_cov(z, parameters, config):
        for zi in (z - parameters.cov.mean[None, :, None, None]):
            zi = zi[:, None] * zi[None, :]
            parameters.cov.cov += zi.sum(3).sum(2)
        return z
    
    @staticmethod
    def layer_adapt(z, parameters, config):
        z = (z - parameters.target.mean[None, :, None, None]).permute(1, 0, 2, 3)
        z = (parameters.W @ z.reshape(z.shape[0], -1)).reshape(z.shape).permute(1, 0, 2, 3)
        return z + parameters.source.mean[None, :, None, None]
    
    @staticmethod
    def transform_shape_parameters(z, cov_phase, config):
        if config.transform_parameters > 0 or config.separate_nodes:
            assert z.shape[1] % 3 == 0 # assume a classic BSP tree using lines as a separator SDF in each inner node
            zs = z.reshape(z.shape[0], -1, 3, *z.shape[2:])
            zs = [zs[:, i] for i in range(zs.shape[1])]
        else:
            zs = [z]
        if config.transform_parameters > 0:
            if cov_phase and config.transform_parameters == 2:
                for i, zi in enumerate(zs):
                    zi = zi.clone()
                    length = torch.linalg.vector_norm(zi[:, :2], ord=2, dim=1)
                    zi[:, 1] = torch.atan2(zi[:, 1], zi[:, 0]) # y coordinate, x coordinate
                    zi[:, 0] = length.clamp(min=10**-12)
                    zs[i] = zi
            else:
                for i, zi in enumerate(zs):
                    zi = zi.clone()
                    length = torch.linalg.vector_norm(zi[:, :2], ord=2, dim=1)
                    length = length[:, None].clamp(min=10**-12)
                    zi[:, :2] /= length
                    zs[i] = torch.cat((length, zi), dim=1)
        if not config.separate_nodes:
            zs = [torch.cat(zs, dim=1)]
        return zs

    @staticmethod
    def shape_mean(z, parameters, config):
        assert config.transform_parameters in (0, 1, 2)
        if hasattr(parameters, "covs") and not parameters.first_cov_call:
            del parameters.covs
        zs = F.transform_shape_parameters(z, False, config)
        if not hasattr(parameters, "covs"):
            parameters.covs = [utils.CovarianceMatrix(zi.shape[1], False) for zi in zs]
            parameters.method = config.method
            parameters.first_cov_call = True
        for zi, cov in zip(zs, parameters.covs):
            cov.add_to_mean(zi.sum(3).sum(2).sum(0))
            cov.num_samples += (zi.shape[0] * zi.shape[2] * zi.shape[3]) - 1
        return z

    @staticmethod
    def shape_cov(z, parameters, config):
        if parameters.first_cov_call and config.transform_parameters > 0:
            for cov in parameters.covs:
                for i in range(1, cov.mean.shape[0], 4):
                    if config.transform_parameters == 1:
                        cov.mean[i:i+2] = nn.functional.normalize(cov.mean[i:i+2], p=2, dim=0)
                    else:
                        assert config.transform_parameters == 2
                        cov.mean[i] = torch.atan2(cov.mean[i+1], cov.mean[i]) # y coordinate, x coordinate
                        cov.mean[i+1] = cov.mean[i+2]
                if config.transform_parameters == 2:
                    cov.mean = torch.cat([cov.mean[i:i+3] for i in range(0, cov.mean.shape[0], 4)], dim=0)
                    cov.cov = cov.cov[:cov.mean.shape[0], :cov.mean.shape[0]]
        parameters.first_cov_call = False
        for zi, cov in zip(F.transform_shape_parameters(z, True, config), parameters.covs):
            zi = zi - cov.mean[None, :, None, None]
            if config.transform_parameters == 2:
                for i in range(1, zi.shape[1], 3):
                    # ensure that difference in angles is in (-pi, pi]
                    delta_angle = zi[:, i]
                    zi[:, i] = torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle))
            for zi in zi:
                zi = zi[:, None] @ zi[None, :]
                cov.cov += zi.sum(3).sum(2)
        return z

    @staticmethod
    def shape_adapt(z, parameters, config):
        zs = []
        for zi, source, W, target in zip(
            F.transform_shape_parameters(z, True, config),
            parameters.source, parameters.Ws, parameters.target
        ):
            zi = zi - target.mean[None, :, None, None]
            if config.transform_parameters == 2:
                for i in range(1, zi.shape[1], 3):
                    # ensure that difference in angles is in (-pi, pi]
                    delta_angle = zi[:, i]
                    zi[:, i] = torch.atan2(torch.sin(delta_angle), torch.cos(delta_angle))
            zi = zi.permute(1, 0, 2, 3)
            zi = (W @ zi.reshape(zi.shape[0], -1)).reshape(zi.shape).permute(1, 0, 2, 3)
            zi = zi + source.mean[None, :, None, None]
            if config.transform_parameters == 1:
                for i in range(0, zi.shape[1], 4):
                    zi[:, i+1:i+3] = nn.functional.normalize(zi[:, i+1:i+3], p=2, dim=1)
                    zi[:, i+1:i+3] *= zi[:, i:i+1]
                zi = torch.cat([zi[:, i:i+3] for i in range(1, zi.shape[1], 4)], dim=1)
            elif config.transform_parameters == 2:
                for i in range(1, zi.shape[1], 3):
                    length, angle = zi[:, i], zi[:, i+1]
                    zi[:, i] = length * torch.cos(angle)
                    zi[:, i+1] = length * torch.sin(angle)
            zs.append(zi)
        return torch.cat(zs, dim=1)


class FeatureTransfer(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        assert len(config.in_decoders.shape) == len(config.in_decoders.content)
        
        self.dataset = core.create_object(datasets, config.source_dataset)
        patch_size = list(self.dataset.test.x.shape[2:])
        patch_info = self.dataset.split_images(*self.dataset.split[:2], patch_size, config.augmentation.phase_one.overlap)
        self.patches = datasets.NormalizedInputProvider(
            self.dataset,
            patch_info,
            self.dataset.base.input_channels,
            patch_size,
            config.augmentation.phase_one.rotate_and_flip
        )
        
        self.model = core.create_object(models, self.config.model, input_shape=self.dataset.training.x.shape[-3:], num_classes=self.dataset.num_classes)
        assert hasattr(self.model, "set_adaptation_hooks")
        print(f"loading model weights from '{config.model_weights}' ...")
        with core.open(config.model_weights, "rb") as f:
            self.model.load_state_dict(torch.load(f, map_location=core.device))
        self.model.eval()
        model_prepare_func = getattr(self.model, "prepare_for_epoch", lambda x,y: None)
        model_prepare_func(*self.config.model_epochs)
        
        num_params = 0
        for params in self.model.parameters():
            num_params += np.product(params.shape)
        self.history["num_model_parameters"] = int(num_params)
        print(f"# of model parameters: {num_params/10**6:.2f}M")
        print() # add an extra new line for nicer output formatting
        
        self.phases = []
        if config.pre_decoders.enabled:
            self.phases.append(0)
        for i, (shape, content) in enumerate(zip(config.in_decoders.shape, config.in_decoders.content)):
            if shape or content:
                self.phases.append(i+1)
        if config.trees.shape.enabled or config.trees.content.enabled:
            self.phases.append(-2)
        if config.output.enabled:
            self.phases.append(-1)
        assert len(self.phases) > 0
        config.epochs = 2 * (len(self.phases) + 1)
        self.num_mini_batches = int(np.ceil(self.patches.shape[0] / self.config.mini_batch_size))
        self.output_set.update(("miou", "mf1", "val_miou", "val_mf1", "test_miou", "test_mf1"))

        config.pre_decoders.partitioning = self.model.feature_partitioning_meta(config.pre_decoders.partitioning)[0]
        self.adaptation_parameters = types.SimpleNamespace(
            config = config,
            base_parameters = types.SimpleNamespace(),
            parameters = {},
            output = types.SimpleNamespace(
                parameters = types.SimpleNamespace(cov=utils.CovarianceMatrix(self.dataset.num_classes, False)),
                learn_mean = config.output.enabled,
                learn_cov = False
            )
        )
        self.model.set_adaptation_hooks(
            "",
            F.base_mean if config.pre_decoders.enabled else None,
            [F.layer_mean if c else None for c in config.in_decoders.shape],
            [F.layer_mean if c else None for c in config.in_decoders.content],
            [F.shape_mean if config.trees.shape.enabled else None, F.layer_mean if config.trees.content.enabled else None],
            self.adaptation_parameters
        )
        
    def pre_train(self, epoch, batch, batch_data):
        if self.eval_params.enabled:
            self.pre_train_eval(epoch, batch, batch_data)
            return
        indices = np.arange(self.patches.shape[0])
        indices = indices[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        batch_data.x = self.patches[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        if self.eval_params.enabled:
            self.train_eval(epoch, batch, batch_data, metrics)
            return
        with torch.no_grad():
            yp = self.model(torch.from_numpy(batch_data.x).to(core.device).requires_grad_(False).float())
            if self.adaptation_parameters.output.learn_mean:
                F.layer_mean(yp, self.adaptation_parameters.output.parameters, self.config.output)
            elif self.adaptation_parameters.output.learn_cov:
                F.layer_cov(yp, self.adaptation_parameters.output.parameters, self.config.output)
        
    def post_epoch(self, epoch, metrics):
        if epoch == 0:
            if hasattr(self.adaptation_parameters.base_parameters, "covs"):
                for cov in self.adaptation_parameters.base_parameters.covs:
                    cov.finalize_mean()
            for parameters in self.adaptation_parameters.parameters.values():
                if hasattr(parameters, "cov"):
                    parameters.cov.finalize_mean()
                else:
                    for cov in parameters.covs:
                        cov.finalize_mean()
            if self.adaptation_parameters.output.learn_mean:
                self.adaptation_parameters.output.parameters.cov.finalize_mean()
                self.adaptation_parameters.output.learn_mean = False
                self.adaptation_parameters.output.learn_cov = True
            self.model.set_adaptation_hooks(
                "",
                F.base_cov if self.config.pre_decoders.enabled else None,
                [F.layer_cov if c else None for c in self.config.in_decoders.shape],
                [F.layer_cov if c else None for c in self.config.in_decoders.content],
                [
                    F.shape_cov if self.config.trees.shape.enabled else None,
                    F.layer_cov if self.config.trees.content.enabled else None
                ],
                self.adaptation_parameters
            )
            return
        elif epoch == 1:
            if hasattr(self.adaptation_parameters.base_parameters, "covs"):
                self.adaptation_parameters.base_parameters.source = []
                for i, cov in enumerate(self.adaptation_parameters.base_parameters.covs):
                    cov.finalize_cov()
                    self.adaptation_parameters.base_parameters.source.append(types.SimpleNamespace(
                        mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                        W = torch.from_numpy(
                            utils.compute_whitening_matrix(cov.cov, cov.corr, self.config.pre_decoders.method)
                        ).to(core.device).float().requires_grad_(False)
                    ))
                    self.adaptation_parameters.base_parameters.covs[i] = utils.CovarianceMatrix(cov.mean.shape[0], False)
            for parameters in self.adaptation_parameters.parameters.values():
                if hasattr(parameters, "cov"):
                    parameters.cov.finalize_cov()
                    parameters.source = types.SimpleNamespace(
                        mean = torch.from_numpy(parameters.cov.mean).to(core.device).float().requires_grad_(False),
                        W = torch.from_numpy(
                            utils.compute_whitening_matrix(
                                parameters.cov.cov, parameters.cov.corr, parameters.method
                            )
                        ).to(core.device).float().requires_grad_(False)
                    )
                    parameters.cov = utils.CovarianceMatrix(parameters.cov.mean.shape[0], False)
                else:
                    parameters.source = []
                    for i, cov in enumerate(parameters.covs):
                        cov.finalize_cov()
                        parameters.source.append(types.SimpleNamespace(
                            mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                            W = torch.from_numpy(
                                utils.compute_whitening_matrix(cov.cov, cov.corr, parameters.method)
                            ).to(core.device).float().requires_grad_(False)
                        ))
                        parameters.covs[i] = utils.CovarianceMatrix(cov.mean.shape[0], False)
            if self.adaptation_parameters.output.learn_cov:
                cov = self.adaptation_parameters.output.parameters.cov
                cov.finalize_cov()
                self.adaptation_parameters.output.parameters.source = types.SimpleNamespace(
                    mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                    W = torch.from_numpy(
                        utils.compute_whitening_matrix(cov.cov, cov.corr, self.config.output.method)
                    ).to(core.device).float().requires_grad_(False)
                )
                self.adaptation_parameters.output.learn_mean = False
                self.adaptation_parameters.output.learn_cov = False
                self.adaptation_parameters.output.parameters.cov = utils.CovarianceMatrix(cov.mean.shape[0], False)
            self.dataset = core.create_object(datasets, self.config.target_dataset)
            patch_size = list(self.dataset.test.x.shape[2:])
            patch_info = self.dataset.split_images(
                *self.dataset.split[:2],
                patch_size,
                self.config.augmentation.phase_one.overlap
            )
            self.patches = datasets.NormalizedInputProvider(
                self.dataset,
                patch_info,
                self.dataset.base.input_channels,
                patch_size,
                self.config.augmentation.phase_one.rotate_and_flip
            )
            self.indices = datasets.IndexMapProvider(
                patch_info,
                patch_size,
                self.config.augmentation.phase_one.rotate_and_flip
            )
            self.num_mini_batches = int(np.ceil(self.patches.shape[0] / self.config.mini_batch_size))
            self.adaptation_parameters.hooks = types.SimpleNamespace(
                base = None,
                shape = [None] * len(self.config.in_decoders.shape),
                content = [None] * len(self.config.in_decoders.content),
                trees = [None, None]
            )
        i = (epoch - 1) // 2
        cov_phase = ((epoch - 1) % 2) == 1
        if i < len(self.phases):
            phase = self.phases[i]
            if phase == 0:
                self.adaptation_parameters.hooks.base = F.base_cov if cov_phase else F.base_mean
                if cov_phase:
                    for cov in self.adaptation_parameters.base_parameters.covs:
                        cov.finalize_mean()
            elif phase == -2:
                if self.config.trees.shape.enabled:
                    self.adaptation_parameters.hooks.trees[0] = F.shape_cov if cov_phase else F.shape_mean
                if self.config.trees.content.enabled:
                    self.adaptation_parameters.hooks.trees[1] = F.layer_cov if cov_phase else F.layer_mean
            elif phase == -1:
                if cov_phase:
                    self.adaptation_parameters.output.parameters.cov.finalize_mean()
                self.adaptation_parameters.output.learn_mean = not cov_phase
                self.adaptation_parameters.output.learn_cov = cov_phase
            else:
                assert phase > 0
                if self.config.in_decoders.shape[phase-1]:
                    self.adaptation_parameters.hooks.shape[phase-1] = F.layer_cov if cov_phase else F.layer_mean
                if self.config.in_decoders.content[phase-1]:
                    self.adaptation_parameters.hooks.content[phase-1] = F.layer_cov if cov_phase else F.layer_mean
            if cov_phase:
                for parameters in self.adaptation_parameters.parameters.values():
                    if hasattr(parameters, "cov"):
                        if parameters.cov.num_samples > 0 and not hasattr(parameters, "target"):
                            parameters.cov.finalize_mean()
                    else:
                        if parameters.covs[0].num_samples > 0 and not hasattr(parameters, "target"):
                            for cov in parameters.covs:
                                cov.finalize_mean()
        if i > 0 and not cov_phase:
            phase = self.phases[i-1]
            if phase == 0:
                self.adaptation_parameters.base_parameters.target = []
                for cov in self.adaptation_parameters.base_parameters.covs:
                    cov.finalize_cov()
                    self.adaptation_parameters.base_parameters.target.append(types.SimpleNamespace(
                        mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                        W = torch.from_numpy(
                            utils.compute_whitening_matrix(cov.cov, cov.corr, self.config.pre_decoders.method)
                        ).to(core.device).float().requires_grad_(False)
                    ))
                self.adaptation_parameters.base_parameters.Ws = []
                for source, target in zip(
                    self.adaptation_parameters.base_parameters.source,
                    self.adaptation_parameters.base_parameters.target
                ):
                    self.adaptation_parameters.base_parameters.Ws.append(torch.linalg.inv(source.W) @ target.W)
                self.adaptation_parameters.hooks.base = F.base_adapt
            elif phase == -2:
                if self.config.trees.shape.enabled:
                    self.adaptation_parameters.hooks.trees[0] = F.shape_adapt
                if self.config.trees.content.enabled:
                    self.adaptation_parameters.hooks.trees[1] = F.layer_adapt
            elif phase == -1:
                cov = self.adaptation_parameters.output.parameters.cov
                cov.finalize_cov()
                self.adaptation_parameters.output.parameters.target = types.SimpleNamespace(
                    mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                    W = torch.from_numpy(
                        utils.compute_whitening_matrix(cov.cov, cov.corr, self.config.output.method)
                    ).to(core.device).float().requires_grad_(False)
                )
                sourceW = self.adaptation_parameters.output.parameters.source.W
                targetW = self.adaptation_parameters.output.parameters.target.W
                self.adaptation_parameters.output.parameters.W = torch.linalg.inv(sourceW) @ targetW
                self.adaptation_parameters.output.learn_mean = False
                self.adaptation_parameters.output.learn_cov = False
            else:
                assert phase > 0
                if self.config.in_decoders.shape[phase-1]:
                    self.adaptation_parameters.hooks.shape[phase-1] = F.layer_adapt
                if self.config.in_decoders.content[phase-1]:
                    self.adaptation_parameters.hooks.content[phase-1] = F.layer_adapt
            for parameters in self.adaptation_parameters.parameters.values():
                if hasattr(parameters, "cov"):
                    if parameters.cov.num_samples > 0 and not hasattr(parameters, "target"):
                        cov = parameters.cov
                        cov.finalize_cov()
                        parameters.target = types.SimpleNamespace(
                            mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                            W = torch.from_numpy(
                                utils.compute_whitening_matrix(cov.cov, cov.corr, parameters.method)
                            ).to(core.device).float().requires_grad_(False)
                        )
                        parameters.W = torch.linalg.inv(parameters.source.W) @ parameters.target.W
                else:
                    if parameters.covs[0].num_samples > 0 and not hasattr(parameters, "target"):
                        parameters.target = []
                        for cov in parameters.covs:
                            cov.finalize_cov()
                            parameters.target.append(types.SimpleNamespace(
                                mean = torch.from_numpy(cov.mean).to(core.device).float().requires_grad_(False),
                                W = torch.from_numpy(
                                    utils.compute_whitening_matrix(cov.cov, cov.corr, parameters.method)
                                ).to(core.device).float().requires_grad_(False)
                            ))
                        parameters.Ws = []
                        for source, target in zip(parameters.source, parameters.target):
                            parameters.Ws.append(torch.linalg.inv(source.W) @ target.W)
        self.model.set_adaptation_hooks(
            "",
            self.adaptation_parameters.hooks.base,
            self.adaptation_parameters.hooks.shape,
            self.adaptation_parameters.hooks.content,
            self.adaptation_parameters.hooks.trees,
            self.adaptation_parameters
        )
        
    def pre_evaluate(self, epoch):
        image_id = self.eval_params.image_id
        print(f"[{datetime.datetime.now()}] evaluating image {image_id+1} ...")
        patch_size = list(self.dataset.test.x.shape[2:])
        patch_info = self.dataset.split_images(image_id, image_id+1, patch_size, self.config.augmentation.phase_two.overlap)
        self.eval_params.x = datasets.NormalizedInputProvider(self.dataset, patch_info, self.dataset.base.input_channels, patch_size, self.config.augmentation.phase_two.rotate_and_flip)
        self.eval_params.i = datasets.IndexMapProvider(patch_info, patch_size, self.config.augmentation.phase_two.rotate_and_flip)
        image = self.dataset.base.images[image_id].base
        self.num_mini_batches = int(np.ceil(self.eval_params.x.shape[0] / self.config.mini_batch_size))
        self.eval_params.logits = np.zeros((*image.shape[:2], self.dataset.num_classes), dtype=np.float64)
        self.eval_params.num_predictions = np.zeros((*image.shape[:2], 1), dtype=np.uint64)
        self.eval_params.time = time.perf_counter()
        
    def pre_train_eval(self, epoch, batch, batch_data):
        indices = np.arange(self.eval_params.x.shape[0])
        indices = indices[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        batch_data.x = self.eval_params.x[indices]
        batch_data.index_map = self.eval_params.i[indices]
    
    def train_eval(self, epoch, batch, batch_data, metrics):
        x = torch.from_numpy(batch_data.x).to(core.device).float().requires_grad_(False)
        yp = self.model(x)
        if self.config.output.enabled:
            yp = F.layer_adapt(yp, self.adaptation_parameters.output.parameters, self.config.output)
        for yp, i in zip(yp.softmax(1), batch_data.index_map):
            if not torch.all(yp.isfinite()):
                continue
            self.eval_params.logits[i[1], i[2]] += yp.permute(1, 2, 0).cpu().numpy()
            self.eval_params.num_predictions[i[1], i[2]] += 1
        
    def post_evaluate(self, epoch):
        self.eval_times.append(time.perf_counter() - self.eval_params.time)
        subset = self.dataset.get_image_subset(self.eval_params.image_id)[0]
        path = f"{core.output_path}/images/{subset}_{self.eval_params.image_id}"
        core.call(f"mkdir -p {path}")
        
        img = self.dataset.base.images[self.eval_params.image_id]
        img, depth, yt = img.base, img.depth, img.gt
        rgb = img[:, :, self.dataset.base.visualization_channels]
        if rgb.dtype != np.uint8:
             rgb = np.asarray(rgb, dtype=np.uint8)
        cv.imwrite(f"{path}/input.png", np.flip(rgb, axis=2), (cv.IMWRITE_PNG_COMPRESSION, 9))
        yt = yt if isinstance(yt, np.ndarray) else yt.get_semantic_image()
        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        cv.imwrite(f"{path}/gt.png", lut[yt], (cv.IMWRITE_PNG_COMPRESSION, 9))
        
        yt = yt if yt.dtype==np.int32 else np.asarray(yt, dtype=np.int32)
        self.eval_params.num_predictions[self.eval_params.num_predictions == 0] = 1
        yp = self.eval_params.logits / self.eval_params.num_predictions
        ypc = np.argmax(yp, axis=2)
        cv.imwrite(f"{path}/prediction.png", lut[ypc], (cv.IMWRITE_PNG_COMPRESSION, 9))
        
        if self.config.save_logits:
            self.logits[f"logits{self.eval_params.image_id}"] = yp.copy().astype(np.float32)
        
        conf_mat = self.conf_mat[subset]
        conf_mat.add(
            np.expand_dims(yt, axis=0),
            np.expand_dims(ypc, axis=0)
        )
        rust.prepare_per_pixel_entropy(yp, 10**-6)
        yp = np.sum(yp, axis=2)
        entropy = self.entropy[subset]
        entropy[0] += np.sum(yp)
        entropy[1] += np.prod(yp.shape)
        
        prefix = f"image{self.eval_params.image_id}_"
        conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        conf_mat.reset()
        conf_mat.add(
            np.expand_dims(yt, axis=0),
            np.expand_dims(ypc, axis=0)
        )
        self.history[f"{prefix}conf_mat"] = conf_mat.to_dict()
        self.history[f"{prefix}entropy"] = float(np.mean(yp))
    
    def finalize(self):
        self.eval_times = []
        self.logits = {}
        self.conf_mat = {
            "training": utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class),
            "validation": utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class),
            "test": utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        }
        for conf_mat in self.conf_mat.values():
            conf_mat.reset()
        self.entropy = {"training": [0, 0], "validation": [0, 0], "test": [0, 0]}
        ignored_subsets = set(getattr(self.config, "ignored_subsets", []))
        for image_id in range(len(self.dataset.base.images)):
            subset = self.dataset.get_image_subset(image_id)
            if ignored_subsets.isdisjoint(set(subset)):
                self.evaluate(self.config.model_epochs[0], image_id=image_id)
            else:
                subset = subset[0]
                n = self.conf_mat[subset].C.shape[0]
                self.conf_mat[subset].C = 1 - np.eye(n, dtype=self.conf_mat[subset].C.dtype)
                self.entropy[subset] = [0, 1]
        self.history["eval_times"] = self.eval_times.copy()
        for prefix, subset in (("", "training"), ("val_", "validation"), ("test_", "test")):
            conf_mat, entropy = self.conf_mat[subset], self.entropy[subset]
            for key, value in conf_mat.compute_metrics().__dict__.items():
                self.history[f"{prefix}{key}"] = value
            self.history[f"{prefix}conf_mat"] = conf_mat.to_dict()
            self.history[f"{prefix}entropy"] = float(entropy[0] / entropy[1])
        if len(self.logits) > 0:
            print(f"[{datetime.datetime.now()}] saving logits ...")
            np.savez_compressed(f"{core.output_path}/logits.npz", **self.logits)
