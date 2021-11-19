import torch.nn as nn

from src.utils import get_feature_detector


def inception() -> nn.Module:
    """Inception model used in StyleGAN for KID calculation

    Returns:
        nn.Module: Inception
    """

    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    model = get_feature_detector(url)
    return model
