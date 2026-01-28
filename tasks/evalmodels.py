import core
import datasets
import models
import numpy as np
import torch
import utils
import time
import rust
import cv2 as cv
import glob
import datetime


class EvalModels(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        
        self.dataset = core.create_object(datasets, config.dataset)
        shape = list(self.dataset.training.x.shape[-3:])
        self.model = core.create_object(models, config.model, input_shape=shape, num_classes=self.dataset.num_classes)
        assert getattr(self.model, "requires_dct_input", 0) == 0
        self.model_prepare_func = getattr(self.model, "prepare_for_epoch", lambda x,y: None)
        num_params = 0
        for params in self.model.parameters():
            num_params += np.product(params.shape)
        self.history["num_model_parameters"] = int(num_params)
        print(f"# of model parameters: {num_params/10**6:.2f}M")
        print() # add an extra new line for nicer output formatting
        
        self.model_weights = tuple(sorted(glob.glob(config.model_weights_path)))
        config.epochs = len(self.model_weights)
        if config.epochs == 0:
            raise RuntimeError(f"no model weights found at '{config.model_weights_path}'")
        self.num_mini_batches = 1
        self.model_weights_epochs = getattr(config, "model_weights_epochs", {})
        
        self.conf_mats = {
            k: utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class) for k in ("train", "val", "test")
        }
        self.output_set.update(("miou", "mf1", "val_miou", "val_mf1", "test_miou", "test_mf1"))
        self.rng = np.random.RandomState(core.random_seeds[0])
        
        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        for image_id, img in enumerate(self.dataset.base.images):
            subset = self.dataset.get_image_subset(image_id)[0]
            path = f"{core.output_path}/images/{subset}_{image_id}"
            core.call(f"mkdir -p {path}")
            rgb = img.base[:,:,self.dataset.base.visualization_channels]
            if rgb.dtype != np.uint8:
                rgb = np.asarray(rgb, dtype=np.uint8)
            cv.imwrite(f"{path}/input.png", np.flip(rgb, axis=2), (cv.IMWRITE_PNG_COMPRESSION, 9))
            cv.imwrite(f"{path}/gt.png", lut[img.gt], (cv.IMWRITE_PNG_COMPRESSION, 9))
    
    def pre_epoch(self, epoch):
        model = self.model_weights[epoch]
        print(f"loading model weights from '{model}' ...")
        with core.open(model, "rb") as f:
            self.model.load_state_dict(torch.load(f, map_location=core.device))
        self.model.eval()
        self.model_name = model.split("/")[-1].split(".")[0]
        if self.model_name in self.model_weights_epochs:
            self.model_prepare_func(*self.model_weights_epochs[self.model_name])
        else:
            self.model_prepare_func(0, 1)
        self.eval_times = []
        for conf_mat in self.conf_mats.values():
            conf_mat.reset()
        self.entropy = {k: [0, 0] for k in self.conf_mats.keys()}
        
    def pre_evaluate(self, epoch):
        print(f"[{datetime.datetime.now()}] evaluating image {self.eval_params.image_id} ...")
        self.eval_params.dataset = self.dataset.get_image_view(self.eval_params.image_id)
        self.num_mini_batches = int(np.ceil(self.eval_params.dataset.x.shape[0] / self.config.mini_batch_size))
        img = self.dataset.base.images[self.eval_params.image_id].base
        self.eval_params.logits = torch.zeros((*img.shape[:2], self.dataset.num_classes), dtype=torch.float64, device=core.device)
        self.eval_params.num_predictions = torch.zeros((*img.shape[:2], 1), dtype=torch.int32, device=core.device)
        self.eval_params.time = time.perf_counter()
        
    def pre_train(self, epoch, batch, batch_data):
        if not self.eval_params.enabled:
            return
        indices = np.arange(self.eval_params.dataset.x.shape[0])
        indices = indices[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        batch_data.x = self.eval_params.dataset.x[indices]
        batch_data.index_map = self.eval_params.dataset.index_map[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            return
        x = torch.from_numpy(batch_data.x).to(core.device).float().requires_grad_(False)
        for yp, i in zip(
            self.model(x).softmax(1),
            torch.from_numpy(batch_data.index_map).to(core.device).long().requires_grad_(False)
        ):
            if not torch.all(yp.isfinite()):
                continue
            self.eval_params.logits[i[1], i[2]] += yp.permute(1, 2, 0)
            self.eval_params.num_predictions[i[1], i[2]] += 1
        
    def post_evaluate(self, epoch):
        self.eval_times.append(time.perf_counter() - self.eval_params.time)
        subset_full, subset = self.dataset.get_image_subset(self.eval_params.image_id)[:2]
        
        yt = self.dataset.base.images[self.eval_params.image_id].gt
        yt = yt if isinstance(yt, np.ndarray) else yt.get_semantic_image()
        yt = yt if yt.dtype == np.int32 else np.asarray(yt, dtype=np.int32)
        self.eval_params.num_predictions[self.eval_params.num_predictions == 0] = 1
        yp = self.eval_params.logits / self.eval_params.num_predictions
        yp, ypc = yp.cpu().numpy(), yp.argmax(2).cpu().numpy()

        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        cv.imwrite(
            f"{core.output_path}/images/{subset_full}_{self.eval_params.image_id}/{self.model_name}.png",
            lut[ypc], (cv.IMWRITE_PNG_COMPRESSION, 9)
        )
        
        if self.config.save_logits:
            self.logits[f"logits{self.eval_params.image_id}"] = yp.copy().astype(np.float32)
        
        conf_mat = self.conf_mats[subset]
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
        self.logits = {}
        per_image_metrics = {}
        for image_id in range(len(self.dataset.base.images)):
            self.evaluate(epoch, image_id=image_id, metrics=per_image_metrics)

        m = {"eval_times": self.eval_times, **per_image_metrics}
        for k, conf_mat in self.conf_mats.items():
            prefix = f"{k}_" if k != "train" else ""
            entropy = self.entropy[k]
            for k2, value in conf_mat.compute_metrics().__dict__.items():
                m[f"{prefix}{k2}"] = value
            m[f"{prefix}conf_mat"] = conf_mat.to_dict()
            m[f"{prefix}entropy"] = float(entropy[0] / entropy[1])
        metrics.__dict__[self.model_name] = m
        
        for k in self.output_set:
            metrics.__dict__[k] = m[k]
        metrics.model_name = self.model_name

        if len(self.logits) > 0:
            print(f"[{datetime.datetime.now()}] saving logits ...")
            np.savez_compressed(f"{core.output_path}/{self.model_name}_logits.npz", **self.logits)
