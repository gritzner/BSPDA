import core
import datasets
import models
import numpy as np
import torch
import utils
import time
import types
import cv2 as cv
import rust
import datetime
from itertools import repeat
import glob


class StyleTransfer(utils.ModelFitter):
    def __init__(self, config, params):
        config.epochs = 1
        super().__init__(config)
        
        self.dataset = core.create_object(datasets, config.source_dataset)
        assert self.dataset.base.images[0].base.dtype == np.uint8
        self.dataset.base.visualization_channels = np.asarray([c for c in sorted(self.dataset.base.input_channels) if c >= 0], dtype=np.int32)
        patch_size = list(self.dataset.test.x.shape[2:])
        patch_info = self.dataset.split_images(*self.dataset.split[:2], patch_size, config.augmentation.phase_one.overlap)
        patches = datasets.InputProvider(self.dataset, patch_info, patch_size, config.augmentation.phase_one.rotate_and_flip)
        patches.no_threading = True

        self.window = np.ones(patch_size, dtype=np.float32)
        if config.fft.windowing.enabled:
            assert patch_size[0] == patch_size[1]
            window = getattr(np, config.fft.windowing.func)
            if config.fft.windowing.func != "kaiser":
                window = window(patch_size[0])
            else:
                window = window(patch_size[0], config.fft.windowing.beta)
            i = np.argmax(window)
            for y in range(patch_size[0]):
                y2 = ((y - i) / i)**2
                for x in range(patch_size[1]):
                    if config.fft.windowing.circular_symmetry:
                        x2 = ((x - i) / i)**2
                        r = np.sqrt(y2 + x2)
                        r = i + r * i
                        r = min(int(r), window.shape[0]-1)
                        self.window[y, x] = window[r]
                    else:
                        self.window[y, x] = window[y] * window[x]
            del window
        
        print(f"\n[{datetime.datetime.now()}] learning source statistics (1/2) ...")
        cov = []
        total_mem = 0
        for z in self.get_samples(0, patches, config.fft):
            total_mem += 8 * (z.shape[0] + 2 * z.shape[0]**2)
            if (total_mem / 1024**3) > config.memory_limit:
                raise RuntimeError(f"attemping to allocate more than {config.memory_limit:.1f} GB memory for covariance matrices")
            cov.append(utils.CovarianceMatrix(z.shape[0], False))
        del total_mem
        yx = types.SimpleNamespace(amp=set(), phase=set())
        for i in self.indices:
            j = i[0] < self.dataset.base.images[0].base.shape[-1]
            for y, x in zip(i[1][j], i[2][j]):
                yx.amp.add((y, x))
            j = np.logical_not(j)
            for y, x in zip(i[1][j], i[2][j]):
                yx.phase.add((y, x))
        self.history["num_fft_yx_samples"]["amplitude"] = len(yx.amp)
        self.history["num_fft_yx_samples"]["phase"] = len(yx.phase)
        del yx
        for z in core.thread_pool.map(self.get_samples, range(patches.shape[0]), repeat(patches), repeat(config.fft)):
            for i, z in enumerate(z):
                cov[i].add_to_mean(z)
        for i in cov:
            i.finalize_mean()
        print(f"[{datetime.datetime.now()}] learning source statistics (2/2) ...")
        for z in core.thread_pool.map(self.get_samples, range(patches.shape[0]), repeat(patches), repeat(config.fft)):
            for i, z in enumerate(z):
                cov[i].add_to_cov(z)
        for i in cov:
            i.finalize_cov()
        print(f"[{datetime.datetime.now()}] computing source normalization parameters ...")
        source_params = []
        for cov in cov:
            mean, cov, corr = cov.mean, cov.cov, cov.corr
            source_params.append(types.SimpleNamespace(
                mean = mean,
                W = np.linalg.inv(utils.compute_whitening_matrix(cov, corr, config.fft.method)),
            ))
        print()
        
        self.dataset = core.create_object(datasets, config.target_dataset)
        assert self.dataset.base.images[0].base.dtype == np.uint8
        true_visualization_channels = self.dataset.base.visualization_channels
        self.dataset.base.visualization_channels = np.asarray([c for c in sorted(self.dataset.base.input_channels) if c >= 0], dtype=np.int32)
        assert self.dataset.base.visualization_channels.shape[0] == patches.shape[1]
        assert all([a == b for a, b in zip(patch_size, self.dataset.test.x.shape[2:])])
        patch_info = self.dataset.split_images(*self.dataset.split[:2], patch_size, config.augmentation.phase_one.overlap)
        patches = datasets.InputProvider(self.dataset, patch_info, patch_size, config.augmentation.phase_one.rotate_and_flip)
        patches.no_threading = True
        
        print(f"\n[{datetime.datetime.now()}] learning target statistics (1/2) ...")
        cov = []
        for z in self.get_samples(0, patches, config.fft):
            cov.append(utils.CovarianceMatrix(z.shape[0], False))
        for z in core.thread_pool.map(self.get_samples, range(patches.shape[0]), repeat(patches), repeat(config.fft)):
            for i, z in enumerate(z):
                cov[i].add_to_mean(z)
        for i in cov:
            i.finalize_mean()
        print(f"[{datetime.datetime.now()}] learning target statistics (2/2) ...")
        for z in core.thread_pool.map(self.get_samples, range(patches.shape[0]), repeat(patches), repeat(config.fft)):
            for i, z in enumerate(z):
                cov[i].add_to_cov(z)
        for i in cov:
            i.finalize_cov()
        print(f"[{datetime.datetime.now()}] computing adaptation parameters ...")
        self.adaptation_params = []
        for cov, source_params in zip(cov, source_params):
            mean, cov, corr = cov.mean, cov.cov, cov.corr
            self.adaptation_params.append(types.SimpleNamespace(
                target_mean = mean,
                W = source_params.W @ utils.compute_whitening_matrix(cov, corr, config.fft.method),
                source_mean = source_params.mean,
            ))
        del source_params
        del cov
        print()
        
        self.ignored_subsets = set(getattr(config, "ignored_subsets", []))
        for image_id in range(len(self.dataset.base.images)):
            if image_id >= self.dataset.split[1] and not self.ignored_subsets.isdisjoint(set(self.dataset.get_image_subset(image_id))):
                continue
            print(f"[{datetime.datetime.now()}] adapting image {image_id+1}/{len(self.dataset.base.images)+1} ...")
            patch_info = self.dataset.split_images(image_id, image_id+1, patch_size, config.augmentation.phase_two.overlap)
            patches = datasets.InputProvider(
                self.dataset, patch_info, patch_size, config.augmentation.phase_two.rotate_and_flip
            )
            patches.no_threading = True
            indices = datasets.IndexMapProvider(patch_info, patch_size, config.augmentation.phase_two.rotate_and_flip)
            indices.no_threading = True
            image = np.zeros((patches.shape[1], *self.dataset.base.images[image_id].base.shape[:2]), dtype=np.float64)
            n_samples = np.zeros(image.shape[1:], dtype=np.int32)
            for i, sample in core.thread_pool.map(self.get_adapted_sample, range(patches.shape[0]), repeat(patches), repeat(config.fft)):
                i = indices[i]
                image[:, i[1], i[2]] += sample
                n_samples[i[1], i[2]] += 1
            n_samples[n_samples == 0] = 1
            image = image / np.expand_dims(n_samples, 0)
            image = np.clip(image, 0, 255).astype(np.uint8)
            self.dataset.base.images[image_id].base = np.moveaxis(image, 0, 2)
        del self.adaptation_params
        
        print(f"[{datetime.datetime.now()}] recomputing normalization parameters ...")
        self.dataset.compute_weights_and_normalization_params(
            core.get_object_meta_info(config.target_dataset)[1].normalization
        )

        self.dataset.base.visualization_channels = true_visualization_channels
        if config.save_dataset:
            print(f"[{datetime.datetime.now()}] saving dataset ...")
            data = {
                "input_channels": self.dataset.base.input_channels,
                "ir_index": self.dataset.base.ir_index,
                "red_index": self.dataset.base.red_index,
                "visualization_channels": self.dataset.base.visualization_channels,
                "num_classes": self.dataset.num_classes,
                "lut": np.asarray(self.dataset.lut, dtype=np.uint8),
                "ignore_class": self.dataset.ignore_class
            }
            subsets = []
            for image_id, img in enumerate(self.dataset.base.images):
                subsets.append(self.dataset.get_image_subset(image_id)[2])
                data[f"img{image_id}"] = img.base
                data[f"depth{image_id}"] = img.depth
                data[f"yt{image_id}"] = img.gt
            data["subsets"] = np.asarray(subsets, dtype=np.uint8)
            np.savez_compressed(f"{core.output_path}/dataset.npz", **data)
        
        print() # add an extra new line for nicer output formatting
        self.num_mini_batches = 1
        self.output_set.update(("miou", "mf1", "val_miou", "val_mf1", "test_miou", "test_mf1"))
        
    def pre_evaluate(self, epoch):
        image_id = self.eval_params.image_id
        print(f"[{datetime.datetime.now()}] evaluating image {image_id+1} ...")
        patch_size = list(self.dataset.test.x.shape[2:])
        test_time_augmentation = self.dataset.test.x.test_time_augmentation
        overlap = .5 if test_time_augmentation else 0
        patch_info = self.dataset.split_images(image_id, image_id+1, patch_size, overlap)
        self.eval_params.x = datasets.NormalizedInputProvider(self.dataset, patch_info, self.dataset.base.input_channels, patch_size, test_time_augmentation)
        self.eval_params.i = datasets.IndexMapProvider(patch_info, patch_size, test_time_augmentation)
        image = self.dataset.base.images[image_id].base
        self.num_mini_batches = int(np.ceil(self.eval_params.x.shape[0] / self.config.mini_batch_size))
        self.eval_params.logits = np.zeros((*image.shape[:2], self.dataset.num_classes), dtype=np.float64)
        self.eval_params.num_predictions = np.zeros((*image.shape[:2], 1), dtype=np.uint64)
        self.eval_params.time = time.perf_counter()
        
    def pre_train(self, epoch, batch, batch_data):
        if not self.eval_params.enabled:
            return
        indices = np.arange(self.eval_params.x.shape[0])
        indices = indices[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        batch_data.x = self.eval_params.x[indices]
        batch_data.index_map = self.eval_params.i[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            return
        x = torch.from_numpy(batch_data.x).to(core.device).float().requires_grad_(False)
        for yp, i in zip(self.model(x).softmax(1), batch_data.index_map):
            if not torch.all(yp.isfinite()):
                continue
            self.eval_params.logits[i[1], i[2]] += yp.permute(1, 2, 0).cpu().numpy()
            self.eval_params.num_predictions[i[1], i[2]] += 1
        
    def post_evaluate(self, epoch):
        self.eval_times.append(time.perf_counter() - self.eval_params.time)
        subset = self.dataset.get_image_subset(self.eval_params.image_id)[0]
        path = f"{core.output_path}/images/{self.eval_params.model_name}/{subset}_{self.eval_params.image_id}"
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
        self.eval_params.metrics[f"{prefix}conf_mat"] = conf_mat.to_dict()
        self.eval_params.metrics[f"{prefix}entropy"] = float(np.mean(yp))
    
    def post_epoch(self, epoch, metrics):
        if len(getattr(self.config, "model", "")) == 0:
            return
        
        self.model = core.create_object(models, self.config.model, input_shape=self.dataset.training.x.shape[-3:], num_classes=self.dataset.num_classes)
        model_prepare_func = getattr(self.model, "prepare_for_epoch", lambda x,y: None)
        model_prepare_func(*self.config.model_epochs)
        
        num_params = 0
        for params in self.model.parameters():
            num_params += np.product(params.shape)
        self.history["num_model_parameters"] = int(num_params)
        print(f"# of model parameters: {num_params/10**6:.2f}M")
        
        self.conf_mat = {
            "training": utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class),
            "validation": utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class),
            "test": utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        }
        
        for weights_fn in glob.iglob(self.config.model_weights_path):
            model_name = weights_fn.split("/")[-1].split(".")[0]
            print(f"[{datetime.datetime.now()}] loading model weights from '{weights_fn}' ...")
            with core.open(weights_fn, "rb") as f:
                self.model.load_state_dict(torch.load(f, map_location=core.device))
            self.model.eval()
        
            self.eval_times = []
            self.logits = {}
            per_image_metrics = {}
            for conf_mat in self.conf_mat.values():
                conf_mat.reset()
            self.entropy = {"training": [0, 0], "validation": [0, 0], "test": [0, 0]}
            for image_id in range(len(self.dataset.base.images)):
                subset = self.dataset.get_image_subset(image_id)
                if self.ignored_subsets.isdisjoint(set(subset)):
                    self.evaluate(epoch, image_id=image_id, model_name=model_name, metrics=per_image_metrics)
                else:
                    subset = subset[0]
                    n = self.conf_mat[subset].C.shape[0]
                    self.conf_mat[subset].C = 1 - np.eye(n, dtype=self.conf_mat[subset].C.dtype)
                    self.entropy[subset] = [0, 1]

            m = {"eval_times": self.eval_times.copy(), **per_image_metrics}
            for prefix, subset in (("", "training"), ("val_", "validation"), ("test_", "test")):
                conf_mat, entropy = self.conf_mat[subset], self.entropy[subset]
                for key, value in conf_mat.compute_metrics().__dict__.items():
                    m[f"{prefix}{key}"] = value
                m[f"{prefix}conf_mat"] = conf_mat.to_dict()
                m[f"{prefix}entropy"] = float(entropy[0] / entropy[1])
            metrics.__dict__[model_name] = m
            
            if len(self.logits) > 0:
                print(f"[{datetime.datetime.now()}] saving logits ...")
                np.savez_compressed(f"{core.output_path}/{model_name}_logits.npz", **self.logits)

    def get_samples(self, i, patches, config):
        image = patches[i].astype(np.float32) * self.window[None, :, :]
        image = np.fft.rfftn(image, axes=(0, 1, 2) if config.across_channels else (1, 2))
        n = image.shape[0]
        image = np.concatenate((np.abs(image), np.angle(image)), axis=0)

        if not hasattr(self, "indices"):
            self.history["num_fft_yx_samples"] = {"upper_limit": image.shape[1] * image.shape[2]}
            assert image.shape[1] % 2 == 0
            i = [], [], []
            for y in range(-image.shape[1]//2, image.shape[1]//2):
                for x in range(image.shape[2]):
                    if y == 0 and x == 0 and not config.amplitude_center:
                        continue
                    r = None
                    if config.sampling_area_shape.lower() == "circle":
                        r = np.sqrt(y**2 + x**2) / (image.shape[2] - 1)
                    elif config.sampling_area_shape.lower() == "square":
                        r = max(abs(y), x) / (image.shape[2] - 1)
                    else:
                        raise RuntimeError(f"unknown sampling area shape: '{config.sampling_area_shape}'")
                    if r <= config.amplitude_radius:
                        for c in range(n):
                            i[0].append(c)
                            i[1].append(y)
                            i[2].append(x)
                    if (y == 0 and x == 0) or r > config.phase_radius:
                        continue
                    for c in range(n):
                        i[0].append(c+n)
                        i[1].append(y)
                        i[2].append(x)
            assert len(i[0]) > 0 and len(i[0]) == len(i[1]) and len(i[0]) == len(i[2])
            self.indices = [np.asarray(i, dtype=np.int32) for i in i]

            if config.separate_amplitude_and_phase:
                i = self.indices[0] < n
                j = np.logical_not(i)
                self.indices = [
                    [self.indices[0][i], self.indices[1][i], self.indices[2][i]],
                    [self.indices[0][j], self.indices[1][j], self.indices[2][j]],
                ]
                if np.count_nonzero(i) == 0:
                    self.indices = [self.indices[1]]
                elif np.count_nonzero(j) == 0:
                    self.indices = [self.indices[0]]
            else:
                self.indices = [self.indices]
            
            if config.per_coefficient:
                i = []
                for indices in self.indices:
                    ys, xs = np.unique(indices[1]), np.unique(indices[2])
                    for y in ys:
                        j = indices[1] == y
                        for x in xs:
                            k = np.logical_and(j, indices[2] == x)
                            if np.count_nonzero(k) == 0:
                                continue
                            i.append((
                                indices[0][k],
                                indices[1][k],
                                indices[2][k],
                            ))
                self.indices = i
            
            self.indices = tuple([tuple(i) for i in self.indices])
            
        return [image[i, j, k] for i, j, k in self.indices]

    def get_adapted_sample(self, i, patches, config):
        image = np.fft.rfftn(patches[i], axes=(0, 1, 2) if config.across_channels else (1, 2))
        n = image.shape[0]
        image = np.concatenate((np.abs(image), np.angle(image)), axis=0)
        for indices, adaptation_params in zip(self.indices, self.adaptation_params):
            v = image[indices[0], indices[1], indices[2]]
            v = (adaptation_params.W @ (v - adaptation_params.target_mean)) + adaptation_params.source_mean
            image[indices[0], indices[1], indices[2]] = v
        image = image[:n] * np.exp(image[n:] * 1j)
        image = np.fft.irfftn(image, axes=(0, 1, 2) if config.across_channels else (1, 2))
        return i, np.clip(image, 0, 255)
