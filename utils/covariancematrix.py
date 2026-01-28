import numpy as np
import torch
import itertools
import types


class CovarianceMatrix():
    def __init__(self, n, channels_last, num_matrices=0):
        if num_matrices > 0:
            self.mean = np.zeros((n, num_matrices), dtype=np.float64)
            self.cov = np.zeros((n, n, num_matrices), dtype=np.float64)
        else:
            self.mean = np.zeros(n, dtype=np.float64)
            self.cov = np.zeros((n, n), dtype=np.float64)
        self.channels_last = channels_last
        self.num_samples = 0
    
    def add_to_mean(self, x0, x1=None):
        if isinstance(x0, torch.Tensor):
            assert x1 is None
            self._add_to_mean_torch(x0)
            return
        if len(x0.shape) in (1, 2):
            assert x1 is None
            assert x0.shape == self.mean.shape
            self.mean[:] += x0
            self.num_samples += 1
        elif len(x0.shape) == 3:
            if self.channels_last:
                x0 = np.moveaxis(x0, 2, 0)
            if x1 is None:
                assert x0.shape[0] == self.mean.shape[0]
                self.mean[:] += np.sum(x0, axis=(1, 2), dtype=np.float64)
            else:
                assert x0.shape[0] == self.mean.shape[0] - 1
                assert len(x1.shape) == 2
                self.mean[:-1] += np.sum(x0, axis=(1, 2), dtype=np.float64)
                self.mean[-1] += np.sum(x1, dtype=np.float64)
            self.num_samples += np.prod(x0.shape[1:])
        else:
            raise RuntimeError(f"unsupported shape: {x0.shape}")
            
    def _add_to_mean_torch(self, x0):
        if not (isinstance(self.mean, torch.Tensor)):
            self.mean = torch.from_numpy(self.mean).to(x0.device)
            self.cov = torch.from_numpy(self.cov).to(x0.device)
        if len(x0.shape) in (1, 2):
            assert x0.shape == self.mean.shape
            self.mean[:] += x0
            self.num_samples += 1
        elif len(x0.shape) == 3:
            if self.channels_last:
                x0 = x0.permute((2, 0, 1))
            assert x0.shape[0] == self.mean.shape[0]
            for i in range(x0.shape[0]):
                self.mean[i] += x0[i].sum(dtype=torch.float64)
            self.num_samples += np.prod(x0.shape[1:])
        else:
            raise RuntimeError(f"unsupported shape: {x0.shape}")
    
    def finalize_mean(self):
        self.mean /= self.num_samples
    
    def add_to_cov(self, x0, x1=None):
        if isinstance(x0, torch.Tensor):
            assert x1 is None
            self._add_to_cov_torch(x0)
            return
        if len(x0.shape) in (1, 2):
            assert x1 is None
            assert x0.shape == self.mean.shape
            v = x0 - self.mean
            self.cov += v[:, None] * v[None, :]
        elif len(x0.shape) == 3:
            if self.channels_last:
                x0 = np.moveaxis(x0, 2, 0)
            if x1 is None:
                assert x0.shape[0] == self.mean.shape[0]
            else:
                assert x0.shape[0] == self.mean.shape[0] - 1
                assert len(x1.shape) == 2
            for i in range(self.cov.shape[0]):
                vi = (x0[i] if i < x0.shape[0] else x1) - self.mean[i]
                for j in range(i, self.cov.shape[1]):
                    vj = (x0[j] if j < x0.shape[0] else x1) - self.mean[j]
                    self.cov[i, j] += np.sum(vi * vj, dtype=np.float64)
        else:
            raise RuntimeError(f"unsupported shape: {x0.shape}")
    
    def _add_to_cov_torch(self, x0):
        if len(x0.shape) in (1, 2):
            assert x0.shape == self.mean.shape
            v = x0 - self.mean
            if v.shape[0] <= 1024:
                self.cov += v[:, None] * v[None, :]
            else:
                ys, xs = itertools.tee(range(0, v.shape[0], 1024))
                for y, x in itertools.product(ys, xs):
                    self.cov[y:y+1024, x:x+1024] += v[y:y+1024, None] * v[None, x:x+1024]
        elif len(x0.shape) == 3:
            if self.channels_last:
                x0 = x0.permute((2, 0, 1))
            assert x0.shape[0] == self.mean.shape[0]
            for i in range(self.cov.shape[0]):
                vi = x0[i] - self.mean[i]
                for j in range(i, self.cov.shape[1]):
                    vj = x0[j] - self.mean[j]
                    self.cov[i, j] += torch.sum(vi * vj, dtype=torch.float64)
        else:
            raise RuntimeError(f"unsupported shape: {x0.shape}")
    
    def finalize_cov(self):
        self.cov /= self.num_samples - 1
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.cpu().numpy()
            assert self.mean.dtype == np.float64
            self.cov = self.cov.cpu().numpy()
            assert self.cov.dtype == np.float64
        n = self.mean.shape[0]
        if len(self.mean.shape) == 1:
            self.corr = np.eye(n, dtype=np.float64)
        else:
            self.corr = np.stack([np.eye(n, dtype=np.float64) for _ in range(self.mean.shape[1])], axis=2)
            assert self.corr.shape == self.cov.shape
        for i in range(1, n):
            for j in range(i):
                self.cov[i, j] = self.cov[j, i]
                self.corr[i, j] = self.cov[i, j] / np.sqrt(self.cov[i, i] * self.cov[j, j], dtype=np.float64)
                self.corr[j, i] = self.corr[i, j]
                        
    def __getitem__(self, i):
        assert len(self.mean.shape) == 2
        return types.SimpleNamespace(
            mean = self.mean[:, i],
            cov = self.cov[:, :, i],
            corr = self.corr[:, :, i]
        )
