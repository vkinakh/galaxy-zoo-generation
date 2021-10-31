from abc import abstractmethod
from typing import Dict, NoReturn, Optional
from PIL import Image
import random

from tqdm import tqdm
import numpy as np

from sklearn.manifold import TSNE

import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from chamferdist import ChamferDistance

from .base_trainer import BaseTrainer
from src.data import GenDataset, get_dataset, infinite_loader, GANDataset
from src.models.fid import get_fid_fn
from src.models import inception_score
from src.utils import PathOrStr
from src.utils import tsne_display_tensorboard


class GeneratorTrainer(BaseTrainer):
    """Abstract class for generator trainers"""

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        super().__init__(config_path, config)

        # mock objects
        self._g_ema = None
        self._encoder = None
        self._classifier = None

        # load sampling dataset, so when sampling we select real labels
        self._sample_ds = self._get_ds()
        self._columns = self._sample_ds.columns

    @abstractmethod
    def train(self):
        """Trains the model
        Main method, used for training the model"""
        pass

    @abstractmethod
    def _load_model(self):
        """Loads model"""
        pass

    @abstractmethod
    def _save_model(self):
        """Saves model to checkpoint"""
        pass

    def evaluate(self) -> NoReturn:
        """Evaluates the model by computing:

        - FID score
        - Inception Score
        - Chamfer distance
        - attribute control accuracy

        and exploring:
        - TSNE of the latent space
        - explore generated images
        - traverse z1, ..., zk latent dimensions
        - epsilon input values
        - epsilon and z1, ..., zk latent dimensions
        """

        fid_score = self._compute_fid_score()
        self._writer.add_scalar('FID', fid_score, 0)

        i_score = self._compute_inception_score()
        self._writer.add_scalar('IS', i_score, 0)

        self._display_output_eps()
        self._explore_y()

        chamfer_dist = self._chamfer_distance()
        self._writer.add_scalar('Chamfer', float(chamfer_dist), 0)

        self._attribute_control_accuracy()

        self._traverse_zk()
        self._explore_eps()
        self._explore_eps_zs()

    def _attribute_control_accuracy(self) -> Dict:
        """Computes attribute control accuracy

        Returns:
            Dict: attribute control accuracy for each label
        """

        n_out = self._config['dataset']['n_out']
        bs = self._config['batch_size']
        diffs = []

        for _ in tqdm(range(n_out)):
            label = self._sample_label(bs)
            label = label.to(self._device)

            with torch.no_grad():
                img = self._g_ema(label)
                h, _ = self._encoder(img)
                pred = self._classifier(h)

            diff = (label - pred) ** 2
            diffs.append(diff.detach().cpu().numpy())

        diffs = np.array(diffs)
        mean_diffs = np.mean(diffs, axis=1)

        result = {}
        for i in n_out:
            result[self._columns[i]] = mean_diffs[i]
        return result

    def _chamfer_distance(self) -> float:
        """Computes Chamfer distance between real and generated data

        Returns:
            float: computed Chamfer distance
        """

        n_batches = 200

        loader = self._get_dl()
        embeddings = []
        # real data embeddings
        for _ in tqdm(range(n_batches)):
            img, _ = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())

        # generated data embeddings
        for _ in tqdm(range(n_batches)):
            label_oh = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label_oh)
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())

        tsne_emb = TSNE(n_components=3).fit_transform(embeddings)
        n = len(tsne_emb)

        tsne_real = tsne_emb[:n//2, ]
        tsne_fake = tsne_emb[n//2:, ]

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_fake).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    def _explore_eps_zs(self) -> NoReturn:
        """Explores generated images with fixed epsilon and x1, ..., xk vectors"""

        traverse_samples = 8
        y = self._sample_label(traverse_samples)

        log_folder = self._writer.checkpoint_folder.parent / 'explore_eps_zs'
        log_folder.mkdir(exist_ok=True, parents=True)

        imgs = []

        for i in range(8):
            zs = self._g_ema.sample_zs(1)
            zs = torch.repeat_interleave(zs, traverse_samples, dim=0)

            eps = self._g_ema.sample_eps(1)
            eps = torch.repeat_interleave(eps, traverse_samples, dim=0)

            with torch.no_grad():
                img = self._g_ema(y, eps, zs).cpu()
                img = torch.cat([_img for _img in img], dim=1)

            imgs.append(img)

        imgs = torch.cat(imgs, dim=2)
        imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(imgs).save(
            log_folder / 'explore_eps_zs.png'
        )

    def _explore_eps(self) -> NoReturn:
        """Explores generated images with fixed epsilon"""

        traverse_samples = 8
        y = self._sample_label(traverse_samples)

        log_folder = self._writer.checkpoint_folder.parent / 'explore_eps'
        log_folder.mkdir(exist_ok=True, parents=True)

        zs = self._g_ema.sample_zs(traverse_samples)
        imgs = []

        for i in range(traverse_samples):
            eps = self._g_ema.sample_eps(1)
            eps = torch.repeat_interleave(eps, traverse_samples, dim=0)

            with torch.no_grad():
                img = self._g_ema(y, eps, zs).cpu()
                img = torch.cat([_img for _img in img], dim=1)

            imgs.append(img)
        imgs = torch.cat(imgs, dim=2)
        imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(imgs).save(
            log_folder / 'traverse_eps.png'
        )

    def _traverse_zk(self) -> NoReturn:
        """Explores generated images by traversing the dimensions in z1, ..., zk vectors"""

        batch_size = self._config['batch_size']

        log_folder = self._writer.checkpoint_folder.parent / 'traverse_zk'
        log_folder.mkdir(exist_ok=True, parents=True)

        traverse_samples = 8
        y = self._sample_label(traverse_samples)

        # generate images
        with torch.no_grad():
            utils.save_image(
                self._g_ema(y),
                log_folder / 'sample.png',
                nrow=int(batch_size ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )

        traverse_range = 4.0
        intermediate_points = 9
        truncation = 0.7

        zs = self._g_ema.sample_zs(traverse_samples, truncation)
        es = self._g_ema.sample_eps(traverse_samples, truncation)
        _, n_layers, n_dim = zs.shape

        offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)

        for i_layer in range(n_layers):
            for i_dim in range(n_dim):
                imgs = []
                for offset in offsets:
                    _zs = zs.clone()
                    _zs[:, i_layer, i_dim] = offset
                    with torch.no_grad():
                        img = self._g_ema(y, es, _zs).cpu()
                        img = torch.cat([_img for _img in img], dim=1)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=2)

                imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
                Image.fromarray(imgs).save(
                    log_folder / f"traverse_L{i_layer}_D{i_dim}.png"
                )

    def _explore_y(self) -> NoReturn:
        """Generates and saves images from random labels"""

        n = 49
        y = self._sample_label(n)

        zs = self._g_ema.sample_zs(n)
        eps = self._g_ema.sample_eps(n)

        with torch.no_grad():
            imgs = self._g_ema(y, eps, zs).cpu()

        imgs = [imgs[i] for i in range(n)]
        imgs = torch.cat(imgs, dim=2)
        imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)

        log_folder = self._writer.checkpoint_folder.parent / 'explore_y'
        log_folder.mkdir(exist_ok=True, parents=True)
        Image.fromarray(imgs).save(log_folder / 'explore_y.png')

    def _display_output_eps(self) -> NoReturn:
        """Displays TSNE of the epsilon from real dataset and from generated dataset"""

        n_classes = self._config['dataset']['n_out']
        n_batches = 200

        loader = self._get_dl()
        labels, embeddings = [], []

        # real data embeddings
        for _ in tqdm(range(n_batches)):
            img, label = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            labels.extend(label.cpu().numpy().tolist())
            embeddings.extend(h.cpu().numpy())

        # generated data embeddings
        for _ in tqdm(range(n_batches)):
            label_oh = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label_oh)
                h, _ = self._encoder(img)

            label = torch.argmax(label_oh, dim=1) + n_classes

            labels.extend(label.cpu().numpy().tolist())
            embeddings.extend(h.cpu().numpy())

        labels = np.array(labels)
        embeddings = np.array(embeddings)

        tsne_emb = TSNE(n_components=2).fit_transform(embeddings)
        img_tsne = tsne_display_tensorboard(tsne_emb, labels, r'T-SNE of the model $\varepsilon$')
        self._writer.add_image('TSNE', img_tsne, 0)

    def _get_fid_data_transform(self):
        """Returns data transform for FID calculation

        Returns:
            data transform
        """

        name = self._config['dataset']['name']
        size = 299  # this size is needed for Inception network

        if name in ['galaxy_zoo']:
            transform = transforms.Compose([
                transforms.CenterCrop(207),
                transforms.Resize((size, size)),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(0.5, 0.5),
            ])

        else:
            raise ValueError('Unsupported dataset')

        return transform

    def _get_data_transform(self):
        """Returns transform for the data, based on the config

        Returns:
            data transform
        """

        name = self._config['dataset']['name']
        size = self._config['dataset']['size']

        if name in ['galaxy_zoo']:
            transform = transforms.Compose([
                transforms.CenterCrop(207),
                transforms.Resize((size, size)),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(0.5, 0.5),
            ])

        else:
            raise ValueError('Unsupported dataset')

        return transform

    def _compute_fid_score(self) -> float:
        """Computes FID score for the dataset

        Returns:
            float: FID score
        """

        name = self._config['dataset']['name']
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']

        transform = self._get_fid_data_transform()
        dataset = GenDataset(name, path, anno, transform=transform)

        fid_func = get_fid_fn(dataset, self._device, len(dataset))
        fid_score = fid_func(self._g_ema)
        return fid_score

    def _compute_inception_score(self) -> float:
        """Computes inception score for the dataset

        Returns:
            float: inception score
        """

        batch_size = self._config['batch_size']
        dataset = GANDataset(self._g_ema, n=100_000)
        score = inception_score(dataset, batch_size=batch_size, resize=True)[0]
        return score

    def _sample_label(self, n: Optional[int] = None) -> torch.Tensor:
        """Samples y label for the dataset

        Args:
            n: number of labels to sample

        Returns:
            torch.Tensor: sampled random label
        """

        if n is None:
            batch_size = self._config['batch_size']
            n = batch_size

        labels = []
        for _ in range(n):
            idx = random.randrange(len(self._sample_ds))
            _, label = self._sample_ds[idx]
            labels.append(torch.from_numpy(label))

        labels = torch.stack(labels).to(self._device)
        return labels

    def _get_ds(self):

        name = self._config['dataset']['name']
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        columns = None if 'columns' not in self._config['dataset'] else self._config['dataset']['columns']

        transform = self._get_data_transform()
        dataset = get_dataset(name, path, anno_file=anno, transform=transform, columns=columns)
        return dataset

    def _get_dl(self):
        batch_size = self._config['batch_size']
        n_workers = self._config['n_workers']

        dataset = self._get_ds()
        loader = infinite_loader(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=n_workers
            )
        )
        return loader
