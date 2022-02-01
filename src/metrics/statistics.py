# Here we put procedures for the statistical investigation and comparison
# of morphological properties of sets of galaxy images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm

import torch
from corner import corner
from chamferdist import ChamferDistance

from .measures import get_morphology_measures_set, Measures, measures_groups


device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_corner(*data, **kwargs):
    d = np.array(*data).T
    print(d.shape)
    fig = corner(d, **kwargs)
    return fig


def plot_corner_measures_group(group: str, measures: Measures, **kwargs):
    """ create corner plot of group of measures

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures_groups)
    measures: dict
        full dict containing all measures
    """
    data = measures.group(group)
    labels = data.keys
    fid = plot_corner(data.numpy(), labels=labels, **kwargs)
    return fid


def compute_distance_point_clouds_chamfer(
        points_source: torch.Tensor,
        points_target: torch.Tensor):
    """ compute the chamfer distance from source_points to target_points

    Parameter
    ---------
    source_points: torch.Tensor
        3D tensor of shape (N_batches, N_points, N_dimensions)
        contains points supposedly close to target points
    target_points: torch.Tensor
        3D tensor of shape (N_batches, N_points, N_dimensions)
        contains points from ground truth

    """
    chamfer_dist = ChamferDistance()
    dist = chamfer_dist(points_source.to(device), points_target.to(device))
    return dist.detach().cpu().item()


compute_distance_point_clouds = compute_distance_point_clouds_chamfer


def compute_distance_measures_group(
        group: str,
        measures_source: Measures,
        measures_target: Measures):
    """ compute distance between points in group of measures

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures_groups)
    measures: dict
        full dict containing all measures
    """
    return compute_distance_point_clouds(
        measures_source.group(group).torch(),
        measures_target.group(group).torch())


def evaluate_generator(dataloader, generator, latent_dim=128, plot=False, plot_path="~/Pictures/",
                       normalize: bool = True):
    """ evaluate galaxy image generator by computing the distance between
        morphology measures obtained from real images and
        morphology measures obtained from images generated from same labels.
        Distance is computed for point clouds for all groups in measures_groups

    Parameters
    ----------
    dataloader: torch.DataLoader
            contains the target dataset
    generator: torch.Module
            contains the generator that transforms latent ant label vectors to galaxy images
    latent_dim: int
            dimension of latent vector required by generator
    plot: boolean
            if True: plot corner plots for all groups

    normalize: boolean
            if True: output images will be renormalized with mean 0.5 and std 0.5
    Output
    ------
    distances: dict
            (chamfer) distance between point clouds of morphological measures for
            the real dataset and the generated counterparts
            for all groups in measures_groups
    """
    # collect measures from dataset and generator
    target = Measures()
    source = Measures()
    for images, labels in tqdm(dataloader):

        if normalize:
            images = (images * 0.5) + 0.5
        measures_target = get_morphology_measures_set(images)
        labels = labels.to(device)
        latent = torch.randn(len(images), latent_dim).to(device)

        images = generator(y=labels, eps=latent)

        if normalize:
            images = (images * 0.5) + 0.5

        measures_generated = get_morphology_measures_set(images)
        target += measures_target
        source += measures_generated

    # calculate distance between groups of point clouds
    distances = {}
    for group in tqdm(measures_groups.keys()):
        distances[group] = compute_distance_measures_group(group, source, target)
        if plot:

            if group == "ellipticity":  # ellipticity is a single measure, corner plot is useless
                continue

            fig = None
            name = 'InfoSCC-GAN'
            fig = plot_corner_measures_group(group, source, color="b", fig=fig, label_kwargs={"fontsize":16})
            fig = plot_corner_measures_group(group, target, fig=fig, color="r")
            fig.suptitle(name, fontsize=20)
            blue_line = mlines.Line2D([], [], color='blue', label='source')
            red_line = mlines.Line2D([], [], color='red', label='target')
            plt.legend(handles=[blue_line, red_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4, fontsize=16)
            plt.savefig(plot_path / f"measures_combined_{group}.png")
            # plot_corner_measures_group(group, source)
            # plt.savefig(plot_path / f"measures_source_{group}.png")
            # plot_corner_measures_group(group, target)
            # plt.savefig(plot_path / f"measures_target_{group}.png")
            # plt.savefig(plot_path / f"measures_combined_{group}.png")
    distances["total"] = compute_distance_point_clouds(source.torch(), target.torch())
    return distances
