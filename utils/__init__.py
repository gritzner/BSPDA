from .modelfitter import ModelFitter
from .lrscheduler import LearningRateScheduler
from .covariancematrix import CovarianceMatrix
from .confusionmatrix import ConfusionMatrix
from .vectorquantization import create_vector_quantization_layer
from .sam import SAM
import numpy as np
import torch
import torch.nn as nn
import types


def hsv2rgb(h, s, v):
    def f(n):
        k = (n + 6*h) % 6
        return v - v*s*max(0,min(k,4-k,1))
    return round(255*f(5)), round(255*f(3)), round(255*f(1))


def relu_wrapper(relu_type):
    if type(relu_type) == list:
        relu_params = relu_type[1:]
        relu_type = relu_type[0]
    else:
        relu_params = []
    relu_func = getattr(nn, relu_type)
    if relu_type == "PReLU" and len(relu_params) > 0 and relu_params[0] != 1:
        return lambda x: relu_func(x)
    else:
        return lambda _: relu_func(*relu_params)

    
def norm_wrapper(norm_type):
    if norm_type == "LayerNormNd":
        return lambda k, affine=True: nn.GroupNorm(1, k, affine=affine)
    elif norm_type == "PixelNorm2d":
        class PixelNorm2d(nn.Module):
            def __init__(self, k, affine):
                super().__init__()
                self.alpha = nn.Parameter(torch.ones((k, 1, 1)))
                if affine:
                    self.beta = nn.Parameter(torch.zeros((k, 1, 1)))
            
            def forward(self, x):
                d = x - x.mean(1, keepdim=True)
                s = d.pow(2).mean(1, keepdim=True)
                x = d / torch.sqrt(s + 10**-6)
                if hasattr(self, "beta"):
                    return self.alpha * x + self.beta
                return self.alpha * x
            
        return PixelNorm2d
    elif norm_type == "GlobalResponseNorm":
        class GlobalResponseNorm(nn.Module):
            def __init__(self, k, affine):
                super().__init__()
                self.alpha = nn.Parameter(torch.ones((1, k, 1, 1)))
                if affine:
                    self.beta = nn.Parameter(torch.zeros((1, k, 1, 1)))
            
            def forward(self, x):
                nx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
                nx = nx / (nx.mean(dim=1, keepdim=True) + 10**-6)
                x = x + self.alpha * (nx * x)
                if hasattr(self, "beta"):
                    return x + self.beta
                return x
            
        return GlobalResponseNorm
    return getattr(nn, norm_type)

def optim_wrapper(optim_type):
    if optim_type == "SAM":
        return SAM
    else:
        return getattr(torch.optim, optim_type)
    
def get_mini_batch_iterator(mini_batch_size, return_indices=False):
    class MiniBatchIterator():
        def __init__(self):
            self.mini_batch_size = mini_batch_size
            self.return_indices = return_indices
        
        def __call__(self, data, *data2):
            if len(data2) > 0:
                for i in data2:
                    assert data.shape[0] == i.shape[0]
            with torch.no_grad():
                for j in range(0, data.shape[0], self.mini_batch_size):
                    batch_data = data[j:j+self.mini_batch_size]
                    if len(data2) == 0 and not self.return_indices:
                        yield batch_data
                    else:
                        if self.return_indices:
                            batch_data = [np.asarray(range(j,j+self.mini_batch_size))[:batch_data.shape[0]], batch_data]
                        else:
                            batch_data = [batch_data]
                        batch_data.extend([
                            i[j:j+self.mini_batch_size] for i in data2
                        ])
                        yield tuple(batch_data)
        
    return MiniBatchIterator()

def get_scheduler(config):
    if isinstance(config, dict):
        config = types.SimpleNamespace(**config)
    
    class Scheduler():
        def __init__(self):
            params = ",".join(config.parameters)
            self.func = eval(f"lambda {params}: {config.func}")
            self.value_range = config.value_range
            self.value = self.value_range[0]
        
        def __call__(self, *params):
            self.value = self.func(*params)
            self.value = (1-self.value)*self.value_range[0] + self.value*self.value_range[1]
        
    return Scheduler()

def compute_whitening_matrix(cov, corr, method):
    method = method.lower().replace(" ","").replace("_","").replace("-","")
    if method in ("mean", "meanonly", "onlymean"):
        return np.eye(cov.shape[0], dtype=np.float64)
    elif method in ("standardscore", "zscore"):
        return np.diag(1 / np.sqrt(np.diag(cov)))
    elif method == "cholesky":
        return np.transpose(np.linalg.cholesky(np.linalg.inv(cov)))
    elif method in ("zca", "mahalanobis", "pca"):
        eig_val, eig_vec = np.linalg.eigh(cov)
        eig_val = np.maximum(eig_val, 10**-12)
        for i in range(eig_vec.shape[0]):
            if eig_vec[i, i] < 0:
                eig_vec[:, i] *= -1 # ensure np.all(np.diag(eig_vec) >= 0) to remove ambiguity
        whitening = np.diag(1 / np.sqrt(eig_val)) @ np.linalg.inv(eig_vec)
        if method[0] != "p":
            whitening = eig_vec @ whitening
        return whitening
    elif method in ("zcacor", "zcacorr", "pcacor", "pcacorr"):
        eig_val, eig_vec = np.linalg.eigh(corr)
        eig_val = np.maximum(eig_val, 10**-12)
        for i in range(eig_vec.shape[0]):
            if eig_vec[i, i] < 0:
                eig_vec[:, i] *= -1 # ensure np.all(np.diag(eig_vec) >= 0) to remove ambiguity
        whitening = np.diag(1 / np.sqrt(eig_val)) @ np.linalg.inv(eig_vec)
        inv_sqrt_V = np.diag(1 / np.sqrt(np.diag(cov)))
        whitening = whitening @ inv_sqrt_V
        if method[0] != "p":
            whitening = eig_vec @ whitening
        return whitening
    raise RuntimeError(f"unknown dataset normalization method: '{config.method}'")
