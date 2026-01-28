import core
import types
import numpy as np
from .SegmentationDataset import SegmentationDataset
from .SegmentationDatasetPatchProviders import *


class AdaptedSegmentationDataset(SegmentationDataset):
    def __init__(self, config, params):
        print(f"loading adapted dataset from '{config.path}' ...")
        if config.path[0] != "/":
            root = core.user.dataset_paths["adapted_root"]
            config.path = f"{root}/{config.path}.npz"
        data = np.load(config.path)
        self.base = types.SimpleNamespace(
            input_channels = data["input_channels"],
            channels = [c for c in data["input_channels"].tolist() if c >= 0],
            ir_index = int(data["ir_index"]),
            red_index = int(data["red_index"]),
            visualization_channels = data["visualization_channels"],
            num_classes = int(data["num_classes"]),
            lut = data["lut"],
            images = []
        )
        self.ignore_class = int(data["ignore_class"])
        logits = np.load(config.logits) if hasattr(config, "logits") else None
        for i, subset in enumerate(data["subsets"]):
            img = types.SimpleNamespace(
                subset = subset,
                order = i,
                base = data[f"img{i}"],
                depth = data[f"depth{i}"],
                gt = data[f"yt{i}"],
                logits = None
            )
            if not (logits is None):
                img.logits = logits[f"logits{i}"].astype(np.float32)
                assert np.all(img.base.shape[:2] == img.logits.shape[:2])
                assert img.logits.shape[-1] == self.base.num_classes
            self.base.images.append(img)
        self.name = config.path.split("/")[-1].split(".")[0]
        self.num_classes = self.base.num_classes
        self.base.lut = tuple([tuple(color) for color in self.base.lut])
        self.lut = self.base.lut
        
        if hasattr(config, "merge"):
            assert all([i == c for i, c in enumerate(self.base.channels)])
            assert not hasattr(config, "logits")
            for path in config.merge:
                subsets, path = path.split(":")
                subsets = [{
                    "train": 0, "training": 0, "val": 1, "validation": 1, "test": 2
                }[subset.strip().lower()] for subset in subsets.split(",")]
                subsets = set(subsets)
                print(f"merging adapted dataset from '{path}' ...")
                if path[0] != "/":
                    root = core.user.dataset_paths["adapted_root"]
                    path = f"{root}/{path}.npz"
                data = np.load(path)
                assert np.all(self.base.input_channels == data["input_channels"])
                assert self.base.ir_index == int(data["ir_index"])
                assert self.base.red_index == int(data["red_index"])
                assert np.all(self.base.visualization_channels == data["visualization_channels"])
                assert self.base.num_classes == int(data["num_classes"])
                for i, subset in enumerate(data["subsets"]):
                    if not (subset in subsets):
                        continue
                    self.base.images.append(types.SimpleNamespace(
                        subset = subset,
                        order = i,
                        base = data[f"img{i}"],
                        depth = data[f"depth{i}"],
                        gt = data[f"yt{i}"],
                        logits = None
                    ))
            self.base.images.sort(key=lambda img: img.subset) # list.sort() [Python 3.0 to 3.13.6, maybe even later versions] is stable, therefore this is deterministic
            for i, img in enumerate(self.base.images):
                img.order = i
                img.base = img.base[:, :, :len(self.base.channels)]
        
        self.split = np.zeros(4, dtype=np.int32)
        self.split[1:] = len(self.base.images)
        self.base.depth_range = np.asarray([np.inf, -np.inf], dtype=np.float32)
        for img in self.base.images:
            self.split[img.subset] = min(self.split[img.subset], img.order)
            del img.subset
            del img.order
            self.base.depth_range[0] = min(self.base.depth_range[0], np.min(img.depth))
            self.base.depth_range[1] = max(self.base.depth_range[0], np.max(img.depth))
            if img.logits is None:
                del img.logits
        self.compute_weights_and_normalization_params(config.normalization)
        rng = np.random.RandomState(core.random_seeds[config.random_seed])
        self.create_augmentation(rng, config)
        
        self.training = types.SimpleNamespace(
            x = NormalizedAugmentedInputProvider(self, self.base.input_channels, False, config),
            x_no_rad_aug = NormalizedAugmentedInputProvider(self, self.base.input_channels, True, config),
            x_raw = RawAugmentedInputProvider(self, config),
            y = AugmentedOutputProvider(self, config),
            mask = MaskProvider(self, config)
        )
        if hasattr(self.base.images[0], "logits"):
            self.training.yp = LogitsProvider(self, config)
        
        val_patch_info = self.split_images(*self.split[1:3], config.patch_size, .5 if config.augmentation.at_test_time else 0)
        self.validation = types.SimpleNamespace(
            x = NormalizedInputProvider(self, val_patch_info, self.base.input_channels, config.patch_size, config.augmentation.at_test_time),
            y = OutputProvider(self, val_patch_info, config.patch_size, config.augmentation.at_test_time),
            index_map = IndexMapProvider(val_patch_info, config.patch_size, config.augmentation.at_test_time)
        )
        
        test_patch_info = self.split_images(*self.split[2:4], config.patch_size, .5 if config.augmentation.at_test_time else 0)
        self.test = types.SimpleNamespace(
            x = NormalizedInputProvider(self, test_patch_info, self.base.input_channels, config.patch_size, config.augmentation.at_test_time),
            y = OutputProvider(self, test_patch_info, config.patch_size, config.augmentation.at_test_time),
            index_map = IndexMapProvider(test_patch_info, config.patch_size, config.augmentation.at_test_time)
        )
        
        for i, (dataset, label) in enumerate(((self.training, "training"), (self.validation, "validation"), (self.test, "test"))):
            num_pixels = sum([np.prod(img.base.shape[:2]) for img in self.base.images[self.split[i]:self.split[i+1]]])
            if num_pixels >= 10**9:
                num_pixels = f"{num_pixels/10**9:.1f}G"
            elif num_pixels >= 10**6:
                num_pixels = f"{num_pixels/10**6:.1f}M"
            elif num_pixels >= 10**3:
                num_pixels = f"{num_pixels/10**3:.1f}k"
            print(f"# of {label} samples (images, pixels): {dataset.x.shape[0]} ({self.split[i+1]-self.split[i]}, {num_pixels})")
        print(f"input shape: {self.training.x.shape[1:]} -> {self.base.images[0].base.dtype}")
        print(f"# of classes: {self.num_classes} ({self.base.images[0].gt.dtype})")
