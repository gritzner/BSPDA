import numpy as np
import core
import cv2 as cv


class AbstractPatchProvider():
    def __getitem__(self, key):
        if isinstance(key, tuple):
            t = type(key[0])
            result = self.extract_patches(key[0])
            result = result.__getitem__((slice(result.shape[0]), *key[1:]))
        else:
            t = type(key)
            result = self.extract_patches(key)
        return result if t==np.ndarray or t==list or t==slice else result[0]
    
    def extract_patches(self, indices):
        if isinstance(indices, np.ndarray) or isinstance(indices, list):
            indices = np.asarray(indices)
            for i, j in enumerate(indices):
                if j < 0:
                    indices[i] += self.shape[0]
        elif isinstance(indices, slice):
            indices = np.asarray(tuple(range(*indices.indices(self.shape[0]))), dtype=np.int32)
        else:
            indices = np.asarray([indices], dtype=np.int32)
            if indices[0] < 0:
                indices[0] += self.x_shape[0]
        assert np.all(indices>=0) and np.all(indices<self.shape[0])

        result = np.empty([indices.shape[0], *self.shape[1:]], self.dtype)
        
        if getattr(self, "no_threading", False):
            if getattr(self, "test_time_augmentation", False):
                for i in range(result.shape[0]):
                    self.extract_patch_proxy(result[i], indices[i])
            else:
                for i in range(result.shape[0]):
                    self.extract_patch(result[i], indices[i])
            return result
        
        futures = []
        if getattr(self, "test_time_augmentation", False):
            for i in range(result.shape[0]):
                futures.append(core.thread_pool.submit(self.extract_patch_proxy, result[i], indices[i]))
        else:
            for i in range(result.shape[0]):
                futures.append(core.thread_pool.submit(self.extract_patch, result[i], indices[i]))
        for future in futures:
            future.result()
            
        return result
    
    def extract_patch_proxy(self, result, index):
        index, augmentation_index = index // 8, index % 8
        self.extract_patch(result, index)
        if augmentation_index > 0:
            if len(result.shape) == 3:
                if augmentation_index % 2 == 1:
                    result[:] = np.moveaxis(cv.flip(np.moveaxis(result, 0, 2), 0), 2, 0)
                augmentation_index //= 2
                if augmentation_index > 0:
                    for i in range(result.shape[0]):
                        result[i] = cv.rotate(result[i], (cv.ROTATE_90_CLOCKWISE,cv.ROTATE_180,cv.ROTATE_90_COUNTERCLOCKWISE)[augmentation_index-1])
            else:
                if augmentation_index % 2 == 1:
                    result[:] = cv.flip(result, 0)
                augmentation_index //= 2
                if augmentation_index > 0:
                    result[:] = cv.rotate(result, (cv.ROTATE_90_CLOCKWISE,cv.ROTATE_180,cv.ROTATE_90_COUNTERCLOCKWISE)[augmentation_index-1])
    
    def compute_augmented_normalized_patch(self, result, img, depth, gt, index):
        self.compute_normalized_patch(result, img, depth, gt)
        if not self.skip_radiometric_augmentation:
            rng = np.random.RandomState(self.augmentation.radiometric.seed[index])
            for i, c in enumerate(self.channels):
                if c == -1:
                    continue
                result[i,:,:] = self.augmentation.radiometric.contrast[index] * result[i,:,:] + self.augmentation.radiometric.brightness[index]
                result[i,:,:] += self.augmentation.radiometric.noise[index] * rng.randn(*result[i,:,:].shape)
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])
        
    def compute_normalized_patch(self, result, img, depth, gt):
        normalized_patch = np.empty((*img.shape[:2], img.shape[2]+1), dtype=np.float32)
        normalized_patch[:, :, :img.shape[2]] = img
        normalized_patch[:, :, -1] = depth
        normalized_patch = normalized_patch - self.normalization_params.mean[None, None, :]
        normalized_patch = np.matmul(
            self.normalization_params.whitening,
            normalized_patch,
            axes = [(-2, -1), (-1, -2), (-1, -2)]
        )
        
        for i, c in enumerate(self.channels):
            if c == -1:
                result[i] = (gt / (.5 * (self.num_classes-1))) - 1
            elif c == -2:
                result[i] = normalized_patch[:, :, -1]
            elif c == -3:
                result[i] = AbstractPatchProvider.ndvi(normalized_patch, self.ir_index, self.red_index)
            else:
                result[i] = normalized_patch[:, :, c]
        
    def compute_raw_patch(self, result, img, depth, gt):
        result[:img.shape[2]] = np.moveaxis(img, 2, 0)
        result[-1] = depth

    def compute_patch(self, result, img, depth, gt):
        for i, c in enumerate(self.channels):
            if c == -1:
                result[i] = gt
            elif c == -2:
                result[i] = (depth - self.depth_range[0]) * 255 / (self.depth_range[1] - self.depth_range[0])
            elif c == -3:
                result[i] = 127.5 * (AbstractPatchProvider.ndvi(img, self.ir_index, self.red_index) + 1)
            else:
                result[i] = img[:,:,c]
    
    @staticmethod
    def get_all_warped_images(img, transform, out_shape, ignore_class):
        base = cv.warpAffine(
            img.base, transform, (out_shape[1],out_shape[0]),
            flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=0
        )
        depth = cv.warpAffine(
            img.depth, transform, (out_shape[1],out_shape[0]),
            flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=0
        )
        gt = img.gt if isinstance(img.gt,np.ndarray) else img.gt.get_semantic_image()
        gt = cv.warpAffine(
            gt, transform, (out_shape[1],out_shape[0]),
            flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=ignore_class
        )
        return base, depth, gt
    
    @staticmethod
    def get_all_cropped_images(img, offsets, out_shape):
        gt = img.gt if isinstance(img.gt,np.ndarray) else img.gt.get_semantic_image()
        return [image[offsets[0]:offsets[0]+out_shape[0],offsets[1]:offsets[1]+out_shape[1]] for image in (img.base, img.depth, gt)]
    
    @staticmethod
    def ndvi(img, ir_index, red_index):
        ir = np.asarray(img[:,:,ir_index], dtype=np.float64)
        red = np.asarray(img[:,:,red_index], dtype=np.float64)
        numerator = ir - red
        denominator = ir + red
        mask = denominator==0
        numerator[mask] = 0
        denominator[mask] = 1
        return numerator / denominator
    
    @staticmethod
    def apply_flips(img, flips):
        offset = 0 if len(img.shape)==2 else 1
        if flips[0] == 1:
            img[:] = np.flip(img, axis=offset+1)
        if flips[1] == 1:
            img[:] = np.flip(img, axis=offset)


class NormalizedAugmentedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, channels, skip_radiometric_augmentation, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.channels = channels
        self.skip_radiometric_augmentation = skip_radiometric_augmentation
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.normalization_params = parent.normalization_params
        self.num_classes = parent.num_classes
        self.ignore_class = parent.ignore_class
    
        self.shape = (
            config.training_samples,
            self.channels.shape[0],
            *config.patch_size
        )
        self.dtype = np.float32
        
    def extract_patch(self, result, index):
        img = self.images[self.augmentation.image[index]]
        img, depth, gt = AbstractPatchProvider.get_all_warped_images(
            img, self.augmentation.transforms[index],
            result.shape[1:], self.ignore_class
        )
        self.compute_augmented_normalized_patch(result, img, depth, gt, index)

        
class RawAugmentedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.channels = np.arange(self.images[0].base.shape[2]+1, dtype=np.int32)
        self.channels[-1] = -2 # depth
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.depth_range = parent.base.depth_range
        self.ignore_class = parent.ignore_class
        
        self.shape = (
            config.training_samples,
            self.channels.shape[0],
            *config.patch_size
        )
        self.dtype = np.float32
        
    def extract_patch(self, result, index):
        img = self.images[self.augmentation.image[index]]
        img, depth, gt = AbstractPatchProvider.get_all_warped_images(
            img, self.augmentation.transforms[index],
            result.shape[1:], self.ignore_class
        )
        self.compute_raw_patch(result, img, depth, gt)
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])


class AugmentedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.channels = parent.base.visualization_channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.depth_range = parent.base.depth_range
        self.ignore_class = parent.ignore_class
        
        self.shape = (
            config.training_samples,
            self.channels.shape[0],
            *config.patch_size
        )
        self.dtype = np.uint8
        
    def extract_patch(self, result, index):
        img = self.images[self.augmentation.image[index]]
        img, depth, gt = AbstractPatchProvider.get_all_warped_images(
            img, self.augmentation.transforms[index],
            result.shape[1:], self.ignore_class
        )
        self.compute_patch(result, img, depth, gt)
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])


class AugmentedOutputProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.ignore_class = parent.ignore_class
        
        self.shape = (
            config.training_samples,
            *config.patch_size
        )
        self.dtype = np.int32
        
    def extract_patch(self, result, index):
        img = self.images[self.augmentation.image[index]]
        img = img.gt if isinstance(img.gt,np.ndarray) else img.gt.get_semantic_image()
        result[:] = cv.warpAffine(
            img, self.augmentation.transforms[index], (result.shape[1],result.shape[0]),
            flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=self.ignore_class
        )
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])


class LogitsProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        
        self.shape = (
            config.training_samples,
            parent.base.num_classes,
            *config.patch_size
        )
        self.dtype = self.images[0].logits.dtype
        
    def extract_patch(self, result, index):
        logits = self.images[self.augmentation.image[index]].logits
        logits = cv.warpAffine(
            logits, self.augmentation.transforms[index], (result.shape[2],result.shape[1]),
            flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=0
        )
        result[:] = np.moveaxis(logits, 2, 0)
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])


class MaskProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        
        self.dummy = [0, 0]
        for img in self.images:
            self.dummy[0] = max(self.dummy[0], img.base.shape[0])
            self.dummy[1] = max(self.dummy[1], img.base.shape[1])
        self.dummy = np.ones(self.dummy, dtype=np.uint8)
        
        self.shape = (
            config.training_samples,
            *config.patch_size
        )
        self.dtype = self.dummy.dtype
        
    def extract_patch(self, result, index):
        height, width = self.images[self.augmentation.image[index]].base.shape[:2]
        result[:] = cv.warpAffine(
            self.dummy[:height,:width], self.augmentation.transforms[index], (result.shape[1],result.shape[0]),
            flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=0
        )
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])


class OracleProvider(AbstractPatchProvider):
    def __init__(self, parent):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        
        for img in self.images:
            img.oracle = np.zeros(img.gt.shape, dtype=np.uint8)
        
        self.shape = tuple(parent.training.y.shape)
        self.dtype = self.images[0].oracle.dtype
        
    def extract_patch(self, result, index):
        oracle = self.images[self.augmentation.image[index]].oracle
        result[:] = cv.warpAffine(
            oracle, self.augmentation.transforms[index], (result.shape[1],result.shape[0]),
            flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT, borderValue=0
        )
        AbstractPatchProvider.apply_flips(result, self.augmentation.flips[index])


class NormalizedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, channels, patch_size, augmentation):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.channels = channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.normalization_params = parent.normalization_params
        self.num_classes = parent.num_classes
        self.test_time_augmentation = augmentation
        
        self.shape = (
            self.patch_info.shape[0] * (8 if augmentation else 1),
            self.channels.shape[0],
            *patch_size
        )
        self.dtype = np.float32
    
    def extract_patch(self, result, index):
        patch_info = self.patch_info[index]
        img, depth, gt = AbstractPatchProvider.get_all_cropped_images(self.images[patch_info[0]], patch_info[1:], result.shape[1:])
        self.compute_normalized_patch(result, img, depth, gt)


class InputProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, patch_size, augmentation):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.channels = parent.base.visualization_channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.depth_range = parent.base.depth_range
        self.test_time_augmentation = augmentation
        
        self.shape = (
            self.patch_info.shape[0] * (8 if augmentation else 1),
            self.channels.shape[0],
            *patch_size
        )
        self.dtype = np.uint8
        
    def extract_patch(self, result, index):
        patch_info = self.patch_info[index]
        img, depth, gt = AbstractPatchProvider.get_all_cropped_images(self.images[patch_info[0]], patch_info[1:], result.shape[1:])
        self.compute_patch(result, img, depth, gt)


class OutputProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, patch_size, augmentation):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.test_time_augmentation = augmentation
        
        self.shape = (
            self.patch_info.shape[0] * (8 if augmentation else 1),
            *patch_size
        )
        self.dtype = np.int32
        
    def extract_patch(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        img = img.gt if isinstance(img.gt,np.ndarray) else img.gt.get_semantic_image()
        result[:] = img[patch_info[1]:patch_info[1]+result.shape[0],patch_info[2]:patch_info[2]+result.shape[1]]

class IndexMapProvider(AbstractPatchProvider):
    def __init__(self, patch_info, patch_size, augmentation):
        self.patch_info = patch_info
        self.test_time_augmentation = augmentation
        
        self.shape = (
            self.patch_info.shape[0] * (8 if augmentation else 1),
            3,
            *patch_size
        )
        self.dtype = np.int32
        
    def extract_patch(self, result, index):
        patch_info = self.patch_info[index]
        result[0] = patch_info[0]
        for i, j in zip((1,2), (1,0)):
            result[i] = np.expand_dims(np.arange(result.shape[i])+patch_info[i], axis=j)
        