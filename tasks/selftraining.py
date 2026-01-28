import core
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import time
import types
import rust
import cv2 as cv


class SelfTraining(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        
        self.dataset = core.create_object(datasets, config.dataset)
        assert hasattr(self.dataset.training, "yp")
        assert 0 < config.num_samples_per_epoch
        assert config.num_samples_per_epoch * config.epochs <= self.dataset.training.x.shape[0]
        self.dataset.add_oracle()
        
        print("computing pseudo ground truth ...")
        pseudo_gt = []
        assert self.dataset.split[0] == 0
        for img in self.dataset.base.images[:self.dataset.split[1]]:
            if config.pseudo_ground_truth.use_gini:
                img = 1 - np.sum(img.logits**2, axis=2)
            else:
                img = 1 - np.max(img.logits, axis=2)
            pseudo_gt.append(img)
        if config.pseudo_ground_truth.topk > 0:
            assert config.pseudo_ground_truth.topk <= 1
            v = np.concatenate([img.flatten() for img in pseudo_gt])
            config.pseudo_ground_truth.threshold = np.quantile(v, config.pseudo_ground_truth.topk)
            del v
        for image_id, img in enumerate(pseudo_gt):
            self.dataset.base.images[image_id].oracle[img <= config.pseudo_ground_truth.threshold] = 1
        del pseudo_gt
        
        shape = list(self.dataset.training.x.shape[-3:])
        self.model = core.create_object(models, config.model, input_shape=shape, num_classes=self.dataset.num_classes)
        assert getattr(self.model, "requires_dct_input", 0) == 0
        self.model_prepare_func = getattr(self.model, "prepare_for_epoch", lambda x,y: None)
        print(f"loading model weights from '{config.model_weights}' ...")
        with core.open(config.model_weights, "rb") as f:
            self.model.load_state_dict(torch.load(f, map_location=core.device))
        
        num_params = 0
        for params in self.model.parameters():
            num_params += np.product(params.shape)
        self.history["num_model_parameters"] = int(num_params)
        print(f"# of model parameters: {num_params/10**6:.2f}M")
        
        print(f"# of training samples per epoch: {config.num_samples_per_epoch}\n") # add an extra new line for nicer output formatting
        self.num_mini_batches = int(np.ceil(config.num_samples_per_epoch / self.config.mini_batch_size))
        
        self.optim = utils.optim_wrapper(config.optimizer.type)(
            self.model.parameters(), lr=self.config.learning_rate.max_value,
            **(config.optimizer.arguments.__dict__ if hasattr(config.optimizer,"arguments") else {})
        )
        assert not hasattr(self.optim, "first_step")
        self.lr_scheduler = utils.LearningRateScheduler(
            self.optim, self.config.learning_rate.min_value,
            self.config.learning_rate.num_cycles, self.config.learning_rate.cycle_length_factor,
            round(self.config.learning_rate.num_iterations_factor * self.num_mini_batches * self.config.epochs)
        )
        
        self.conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        self.output_set.update(("loss", "acc", "miou", "mf1", "val_loss", "val_acc", "val_miou", "val_mf1"))
        self.rng = np.random.RandomState(core.random_seeds[self.config.shuffle_seed])
        self.full_shuffle_map = self.rng.permutation(self.dataset.training.x.shape[0])
        self.shuffle_map_offset = 0
        
        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        for image_id in np.unique(self.dataset.validation.x.patch_info[:,0]):
            path = f"{core.output_path}/val_images/{image_id}"
            core.call(f"mkdir -p {path}")
            img = self.dataset.base.images[image_id].base[:,:,self.dataset.base.visualization_channels]
            if img.dtype != np.uint8:
                img = np.asarray(img, dtype=np.uint8)
            cv.imwrite(f"{path}/input.png", np.flip(img, axis=2), (cv.IMWRITE_PNG_COMPRESSION, 9))
            cv.imwrite(f"{path}/gt.png", lut[self.dataset.base.images[image_id].gt], (cv.IMWRITE_PNG_COMPRESSION, 9))
    
    def pre_epoch(self, epoch):
        self.model.train()
        self.model_prepare_func(epoch, self.config.epochs)
        self.shuffle_map = self.full_shuffle_map[self.shuffle_map_offset:self.shuffle_map_offset+self.config.num_samples_per_epoch]
        self.shuffle_map_offset += self.config.num_samples_per_epoch
        assert self.shuffle_map_offset <= self.dataset.training.x.shape[0]
        for i in range(self.shuffle_map.shape[0]):
            valid_area = 0
            while valid_area < self.config.min_valid_area_per_sample:
                valid_area = self.dataset.training.oracle[self.shuffle_map[i]]
                valid_area = np.count_nonzero(valid_area == 1) / np.prod(valid_area.shape)
                if valid_area < self.config.min_valid_area_per_sample:
                    self.shuffle_map[i] = self.shuffle_map_offset
                    self.shuffle_map_offset += 1
                    assert self.shuffle_map_offset <= self.dataset.training.x.shape[0]
        self.config.pseudo_ground_truth.dynamic_override.enabled = epoch >= self.config.pseudo_ground_truth.dynamic_override.epoch_delay
        self.conf_mat.reset()
        
    def pre_evaluate(self, epoch):
        self.model.eval()
        self.model_prepare_func(epoch, self.config.epochs)
        self.num_mini_batches = int(np.ceil(self.eval_params.dataset.x.shape[0] / self.config.mini_batch_size))
        self.shuffle_map = np.arange(self.eval_params.dataset.x.shape[0], dtype=np.int32)
        self.eval_params.loss = 0
        self.eval_params.predictions = {
            image_id: [
                np.zeros((*self.eval_params.images[image_id].base.shape[:2], self.dataset.num_classes), dtype=np.float64),
                np.zeros((*self.eval_params.images[image_id].base.shape[:2], 1), dtype=np.uint64)
            ] for image_id in np.unique(self.eval_params.dataset.x.patch_info[:,0])
        }
        self.eval_params.time = time.perf_counter()
        
    def pre_train(self, epoch, batch, batch_data):
        indices = self.shuffle_map[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        dataset = self.eval_params.dataset if self.eval_params.enabled else self.dataset.training
        batch_data.x = dataset.x[indices]
        batch_data.yt = dataset.y[indices]
        if not self.eval_params.enabled:
            batch_data.target_yp = dataset.yp[indices]
            batch_data.mask = dataset.oracle[indices]
            batch_data.target_yp_mask = dataset.mask[indices]
        else:
            batch_data.index_map = dataset.index_map[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        inputs = {
            "x": torch.from_numpy(batch_data.x).to(core.device).float().requires_grad_(),
            "ce_loss_func": nn.functional.cross_entropy,
            "weight": self.dataset.class_weights,
            "ignore_index": self.dataset.ignore_class
        }
        if not self.eval_params.enabled:
            self.optim.zero_grad()
            target_yp = torch.from_numpy(batch_data.target_yp).to(core.device).float().detach()
            _, pseudo_yt = target_yp.max(1)
            pseudo_yt[torch.from_numpy(batch_data.mask).to(core.device).detach() == 0] = self.dataset.ignore_class
            pseudo_yt_mask = torch.from_numpy(batch_data.target_yp_mask).to(core.device).detach()
            
            i = epoch * self.num_mini_batches + batch
            if i < len(self.config.active_learning.iterations) or self.config.pseudo_ground_truth.dynamic_override.enabled:
                with torch.no_grad():
                    yp = self.model(**{k: (v.detach() if k == "x" else v) for k, v in inputs.items()}).detach().softmax(1)
                    if self.config.pseudo_ground_truth.dynamic_override.enabled:
                        if self.config.pseudo_ground_truth.use_gini:
                            ypc = yp.argmax(1)
                            yp2 = 1 - (yp**2).sum(1)
                        else:
                            yp2, ypc = yp.max(1)
                            yp2 = 1 - yp2
                        if self.config.pseudo_ground_truth.dynamic_override.topk > 0:
                            assert self.config.pseudo_ground_truth.dynamic_override.topk <= 1
                            self.config.pseudo_ground_truth.dynamic_override.threshold = yp.quantile(self.config.pseudo_ground_truth.dynamic_override.topk)
                        j = yp2 <= self.config.pseudo_ground_truth.dynamic_override.threshold
                        n = torch.logical_and(ypc[j] != pseudo_yt[j], pseudo_yt_mask[j] == 1).count_nonzero().item()
                        metrics.updated_pseudo_ground_truth_pixels = n
                        metrics.updated_pseudo_ground_truth_max_pixels = (pseudo_yt_mask == 1).count_nonzero().item()
                        metrics.updated_pseudo_ground_truth_relative = n / metrics.updated_pseudo_ground_truth_max_pixels
                        pseudo_yt[j] = ypc[j]
                    else:
                        metrics.updated_pseudo_ground_truth_pixels = 0
                        metrics.updated_pseudo_ground_truth_max_pixels = 0
                        metrics.updated_pseudo_ground_truth_relative = 0
                    
                    if i < len(self.config.active_learning.iterations):
                        n = self.config.active_learning.iterations[i]
                        assert n <= pseudo_yt.shape[0]
                        if self.config.active_learning.random:
                            i = self.rng.permutation(yp.shape[0])[:n]
                        else:
                            if self.config.active_learning.use_gini:
                                yp = 1 - (yp**2).sum(1)
                            else:
                                yp, _ = yp.max(1)
                                yp = 1 - yp
                            yp = yp.reshape(yp.shape[0], -1).quantile(self.config.active_learning.quantile, dim=1)
                            _, i = yp.sort()
                            i = i[-n:]
                        yt = torch.from_numpy(batch_data.yt).to(core.device).long().detach()
                        pseudo_yt[i] = yt[i]
            else:
                metrics.updated_pseudo_ground_truth_pixels = 0
                metrics.updated_pseudo_ground_truth_max_pixels = 0
                metrics.updated_pseudo_ground_truth_relative = 0
            
            pseudo_yt[pseudo_yt_mask == 0] = self.dataset.ignore_class
            inputs["yt"] = pseudo_yt
            yp, loss = self.model(**inputs)
            loss.backward()
            if self.config.gradient_clipping > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            self.optim.step()
            metrics.learning_rate = self.lr_scheduler.get_lr()
            self.lr_scheduler.step(epoch * self.num_mini_batches + batch)
            metrics.loss = loss.item()
            batch_data.yp = yp.argmax(1).cpu().numpy()
        else:
            inputs["yt"] = torch.from_numpy(batch_data.yt).to(core.device).long().detach()
            yp, loss = self.model(**inputs)
            self.eval_params.loss += loss.item()
            batch_data.yp = yp.softmax(1).cpu().numpy()
        
    def post_train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            self.conf_mat.add(batch_data.yt, batch_data.yp)
            current_metrics = self.conf_mat.compute_metrics()
            metrics.acc = current_metrics.acc
            metrics.miou = current_metrics.miou
            metrics.mf1 = current_metrics.mf1
        else:
            for yp, index_map in zip(batch_data.yp, batch_data.index_map):
                p = self.eval_params.predictions[index_map[0,0,0]]
                p[0][index_map[1],index_map[2]] += np.moveaxis(yp, 0, 2)
                p[1][index_map[1],index_map[2]] += 1
        
    def post_evaluate(self, epoch):
        self.eval_params.time = time.perf_counter() - self.eval_params.time
        metrics = self.eval_params.metrics.__dict__
        metrics[f"{self.eval_params.prefix}time"] = self.eval_params.time
        metrics[f"{self.eval_params.prefix}time_per_sample"] = self.eval_params.time / self.eval_params.dataset.x.shape[0]
        metrics[f"{self.eval_params.prefix}loss"] = self.eval_params.loss / self.num_mini_batches
        
        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        self.conf_mat.reset()
        for image_id, p in self.eval_params.predictions.items():
            yt = self.eval_params.images[image_id].gt
            yt = yt if isinstance(yt,np.ndarray) else yt.get_semantic_image()
            yt = yt if yt.dtype==np.int32 else np.asarray(yt, dtype=np.int32)
            yp = p[0] / p[1]
            ypc = np.argmax(yp, axis=2)
            rust.prepare_per_pixel_entropy(yp, 10**-6)
            p[0] = np.sum(yp, axis=2)
            self.conf_mat.add(
                np.expand_dims(yt, axis=0),
                np.expand_dims(ypc, axis=0)
            )
            if hasattr(self.eval_params, "image_prefix"):
                cv.imwrite(
                    f"{core.output_path}/val_images/{image_id}/{self.eval_params.image_prefix}prediction.png",
                    lut[ypc], (cv.IMWRITE_PNG_COMPRESSION, 9)
                )
            prefix = f"image{image_id}_"
            conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
            conf_mat.reset()
            conf_mat.add(
                np.expand_dims(yt, axis=0),
                np.expand_dims(ypc, axis=0)
            )
            metrics[f"{prefix}conf_mat"] = conf_mat.to_dict()
            metrics[f"{prefix}entropy"] = float(np.mean(p[0]))
        
        entropy = [0, 0]
        for p, _ in self.eval_params.predictions.values():
            entropy[0] += np.sum(p)
            entropy[1] += np.prod(p.shape)
        metrics[f"{self.eval_params.prefix}entropy"] = float(entropy[0] / entropy[1])
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            metrics[f"{self.eval_params.prefix}{key}"] = value
        metrics[f"{self.eval_params.prefix}conf_mat"] = self.conf_mat.to_dict()
        
    def post_epoch(self, epoch, metrics):
        epoch_prefix = f"{epoch:03d}_"
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            if key == "acc" or key == "miou" or key == "mf1":
                continue
            metrics.__dict__[key] = value
        metrics.conf_mat = self.conf_mat.to_dict()

        self.evaluate(
            epoch, metrics=metrics, dataset=self.dataset.validation, images=self.dataset.base.images,
            prefix="val_", image_prefix=epoch_prefix
        )
        
        if not (self.config.model_filename_extension is None) and len(self.config.model_filename_extension) > 0:
            path = f"{core.output_path}/{epoch_prefix}model_{metrics.val_miou:.4f}_{metrics.val_mf1:.4f}{self.config.model_filename_extension}"
            print(f"saving model weights to '{path}'")
            with core.open(path, "wb") as f:
                torch.save(self.model.state_dict(), f)
        
        if epoch == 0:
            return
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 28))
        for i, metric in enumerate(("loss", "acc", "miou", "mf1")):
            for j, prefix in enumerate(("", "val_")):
                key = f"{prefix}{metric}"
                axes[i, j].set_title(key)
                data = np.asarray(self.history[key])
                if j == 0:
                    data = data.reshape((epoch+1, self.num_mini_batches))
                    if i == 0:
                        data = np.mean(data, axis=1)
                    else:
                        data = data[:,-1]
                else:
                    data = np.concatenate([data, [metrics.__dict__[key]]])
                axes[i, j].plot(data)
                for k in range(1, data.shape[0]):
                    data[k] = (1 - self.config.smoothing)*data[k] + self.config.smoothing*data[k-1]
                axes[i, j].plot(data)
        plt.tight_layout()
        fig.savefig(f"{core.output_path}/history.pdf")
        plt.close(fig)
        
    def finalize(self):
        self.history["test"] = {}
        for i in range(self.dataset.split[0], self.dataset.split[2]):
            self.dataset.base.images[i] = None
        
        relevant_epochs = set()
        relevant_epochs.add(len(self.history["val_loss"])-1)
        relevant_epochs.add(np.argmin(self.history["val_loss"]))
        relevant_epochs.add(np.argmin(self.history["val_entropy"]))
        relevant_epochs.add(np.argmax(self.history["val_acc"]))
        relevant_epochs.add(np.argmax(self.history["val_miou"]))
        relevant_epochs.add(np.argmax(self.history["val_mf1"]))
        
        import glob
        for model_fn in sorted(glob.iglob(f"{core.output_path}/*_model_*{self.config.model_filename_extension}")):
            epoch = int(model_fn.split("/")[-1].split("_")[0])
            if not epoch in relevant_epochs:
                if self.config.delete_irrelevant_models:
                    core.call(f"rm {model_fn}")
                continue
            print(f"evaluating '{model_fn}' on test set ...")
            with core.open(model_fn, "rb") as f:
                self.model.load_state_dict(torch.load(f, map_location=core.device))
            metrics = types.SimpleNamespace()
            self.evaluate(epoch, metrics=metrics, dataset=self.dataset.test, images=self.dataset.base.images, prefix="")
            self.history["test"][epoch] = metrics.__dict__
