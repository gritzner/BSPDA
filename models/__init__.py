from .SegNeXt import SegNeXtFeatureExtractor, SegNeXt
from .TIMM import TIMMFeatureExtractor, TIMMViTFeatureExtractor
from .UNet import UNet
from .FCN import FCN
from .RAFCN import RAFCN
from .FarSeg import FarSeg
from .PFNet import PFNet
from .DeepLabv3p import DeepLabv3p
from .SegForestNet import SegForestNet
from .ForestEnsemble import ForestEnsemble
from .ResidualModel import ResidualModel
from .WeightedEnsembleModel import WeightedEnsembleModel
from .MixtureOfExpertsModel import MixtureOfExpertsModel


def feature_extractor(config, in_channels):
    if hasattr(config, "backbone"):
        if getattr(config, "vit", False):
            return TIMMViTFeatureExtractor(config, in_channels)
        return TIMMFeatureExtractor(config, in_channels)
    else:
        return SegNeXtFeatureExtractor(config, in_channels)
