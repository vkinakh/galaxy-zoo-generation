"""
This file provides a function to evaluate the distribution of latent variables from a set of images.
Encoder from VAE has to be trained on similar kinds of images
"""

from functools import partial
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import autoencoder
from . import wasserstein
from . import evaluate_cluster_distribution


def evaluate_latent_distribution(generator: nn.Module,
                                 data_loader_test: DataLoader,
                                 data_loader_valid: DataLoader,
                                 encoder: autoencoder.Encoder,
                                 name: str,
                                 N_cluster: int = 10,
                                 batch_size: int = 128):
    """ evaluate the reduced latent distribution of images generated by several models"""

    data_reference = get_latent(data_loader_test, encoder, rescale=False)
    data_validation = get_latent(data_loader_valid, encoder, rescale=False)  # validation set is used to obtain ground truth.
    distribution_evaluation = evaluate_cluster_distribution.DistributionEvaluation(data_reference, N_cluster)
    distribution_evaluation.add("ground truth", data_validation)
    results_wasserstein = {}
    wasser = partial(wasserstein.wasserstein, blur=0.005, scaling=0.95, splits=4)
    results_wasserstein["ground truth"] = wasser(data_validation, data_reference)

    data_generated = get_latent(image_generator(generator, data_loader_valid, batch_size=batch_size),
                                encoder,
                                rescale=False)
    del generator

    distribution_evaluation.add(name, data_generated)
    results_wasserstein[name] = wasser(data_generated, data_reference)
    distribution_evaluation.process(True)

    results_clusters = {
        "histograms": distribution_evaluation.histograms,
        "errors": distribution_evaluation.get_errors(),
        "distances": distribution_evaluation.get_distances(combined=True),
    }
    return results_clusters, results_wasserstein


@torch.no_grad()
def get_latent(dataloader,  # iterable generator that provides a tuple (images, dummy).
               encoder: autoencoder.Encoder,
               rescale: bool = True,  # if True: rescale image from (0,-1) to (-1,1)
               ):
    """ obtain latent vectors for all images in dataloader """
    latent = []
    for images, _ in dataloader:
        images = images.cuda()
        if rescale:
            images = images*2 - 1
        mu, sigma = encoder(images)
        latent.extend(mu.detach().cpu().numpy())
    return np.array(latent)


@torch.no_grad()
def image_generator(model,  # pytorch generator model that takes latent vector and label vector as input
                    dataloader: DataLoader,  # dataloader to provide labels for the predicted distribution
                    batch_size: int
                    ):
    for _, labels in tqdm(dataloader, desc=f"generate images {type(model).__name__}"):
        images = model.sample(batch_size)
        yield images, labels
