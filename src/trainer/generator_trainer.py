from abc import abstractmethod
from typing import Dict, Optional
from PIL import Image
import random

from tqdm import trange
import numpy as np

from sklearn.manifold import TSNE

import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from chamferdist import ChamferDistance
from geomloss import SamplesLoss

from .base_trainer import BaseTrainer
from src.data import get_dataset, infinite_loader, GANDataset
from src.data.dataset_updated import MakeDataLoader
from src.models.fid import get_fid_fn, load_patched_inception_v3
from src.models import inception_score
from src.models import vgg16
from src.utils import PathOrStr
from src.utils import tsne_display_tensorboard
from src.utils import calculate_frechet_distance, slerp


torch.backends.cudnn.benchmark = True


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

    def evaluate(self) -> None:
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
        chamfer_dist = self._compute_chamfer_distance()
        self._writer.add_scalar('Chamfer', float(chamfer_dist), 0)

        ssl_fid = self._compute_ssl_fid()
        self._writer.add_scalar('SSL_FID', ssl_fid, 0)

        ssl_ppl = self._compute_ssl_ppl()
        self._writer.add_scalar('SSL_PPL', ssl_ppl, 0)

        vgg_ppl = self._compute_vgg16_ppl()
        self._writer.add_scalar('VGG_PPL', vgg_ppl, 0)

        kid_inception = self._compute_inception_kid()
        self._writer.add_scalar('KID_Inception', kid_inception, 0)

        kid_ssl = self._compute_ssl_kid()
        self._writer.add_scalar('KID_SSL', kid_ssl, 0)

        self._attribute_control_accuracy()

        self._traverse_zk()
        self._explore_eps()
        self._explore_eps_zs()

    @torch.no_grad()
    def _compute_ssl_kid(self) -> float:
        """Computes Kernel Inception Distance using features computed using pretrained SimCLR

        Returns:
            float: KID
        """

        loader = self._get_dl()

        bs = self._config['batch_size']
        num_samples = 50_000
        n_batches = int(num_samples / bs) + 1

        real_features = []
        for _ in trange(n_batches):
            img, _ = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            real_features.extend(h.cpu().numpy())
        real_features = np.array(real_features)

        gen_features = []
        for _ in trange(n_batches):
            label = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label)
                h, _ = self._encoder(img)
            gen_features.extend(h.cpu().numpy())
        gen_features = np.array(gen_features)

        m = 1000  # max subset size
        num_subsets = 100

        n = real_features.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
            y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return kid

    @torch.no_grad()
    def _compute_inception_kid(self) -> float:
        """Computes Kernel Inception Distance using features computed using pretrained InceptionV3

        Returns:
            float: KID (lower - better)
        """

        encoder = load_patched_inception_v3().to(self._device).eval()

        loader = self._get_dl()

        bs = self._config['batch_size']
        num_samples = 20_000
        n_batches = int(num_samples / bs) + 1

        real_features = []
        for _ in trange(n_batches):
            img, _ = next(loader)
            img = img.to(self._device)

            if img.shape[2] != 299 or img.shape[3] != 299:
                img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

            with torch.no_grad():
                feature = encoder(img)[0].flatten(start_dim=1)
                real_features.extend(feature.cpu().numpy())
        real_features = np.array(real_features)

        gen_features = []
        for _ in trange(n_batches):
            label = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label)

                if img.shape[2] != 299 or img.shape[3] != 299:
                    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

                feature = encoder(img)[0].flatten(start_dim=1)
                gen_features.extend(feature.cpu().numpy())
        gen_features = np.array(gen_features)

        m = 1000  # max subset size
        num_subsets = 100

        n = real_features.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
            y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def _compute_vgg16_ppl(self) -> float:
        """Computes perceptual path length using features computed using pretrained VGG16

        Returns:
            float: perceptual path length (smaller better)
        """

        encoder = vgg16().to(self._device).eval()

        bs = self._config['batch_size']
        num_samples = 50_000
        epsilon = 1e-4
        n_batches = int(num_samples / bs) + 1

        dist = []
        for _ in trange(n_batches):
            label = self._sample_label()
            labels_cat = torch.cat([label, label])

            t = torch.rand([label.shape[0]], device=label.device)
            eps0 = self._g_ema.sample_eps(bs)
            eps1 = self._g_ema.sample_eps(bs)

            z0 = z1 = self._g_ema.sample_zs(bs)

            epst0 = slerp(eps0, eps1, t.unsqueeze(1))
            epst1 = slerp(eps0, eps1, t.unsqueeze(1) + epsilon)

            with torch.no_grad():
                img = self._g_ema(labels_cat, torch.cat([epst0, epst1]), torch.cat([z0, z1]))
                if img.shape[2] != 256 or img.shape[3] != 256:
                    img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bicubic')

                lpips0, lpips1 = encoder(img).chunk(2)

            d = (lpips0 - lpips1).square().sum(1) / epsilon ** 2
            dist.extend(d.cpu().numpy())

        dist = np.array(dist)
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return ppl

    @torch.no_grad()
    def _compute_ssl_ppl(self) -> float:
        """Computes perceptual path length using features computed using SimCLR

        Returns:
            float: perceptual path length (smaller better)
        """

        bs = self._config['batch_size']
        num_samples = 50_000
        epsilon = 1e-4
        n_batches = int(num_samples / bs) + 1

        dist = []
        for _ in trange(n_batches):
            label = self._sample_label()
            labels_cat = torch.cat([label, label])

            t = torch.rand([label.shape[0]], device=label.device)
            eps0 = self._g_ema.sample_eps(bs)
            eps1 = self._g_ema.sample_eps(bs)

            z0 = z1 = self._g_ema.sample_zs(bs)

            epst0 = slerp(eps0, eps1, t.unsqueeze(1))
            epst1 = slerp(eps0, eps1, t.unsqueeze(1) + epsilon)

            with torch.no_grad():
                img = self._g_ema(labels_cat, torch.cat([epst0, epst1]), torch.cat([z0, z1]))
                h, _ = self._encoder(img)

            h0, h1 = h.chunk(2)
            d = (h0 - h1).square().sum(1) / epsilon ** 2
            dist.extend(d.cpu().numpy())

        dist = np.array(dist)
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()

        return ppl

    @torch.no_grad()
    def _compute_ssl_fid(self) -> float:
        """Computes FID on SSL features

        Returns:
            float: FID
        """
        n_samples = 50_000
        bs = self._config['batch_size']
        n_batches = int(n_samples / bs) + 1
        dl = self._get_dl()

        # compute activations
        activations_real = []
        activations_fake = []

        for _ in trange(n_batches):
            img, lbl = next(dl)
            img = img.to(self._device)
            lbl = lbl.to(self._device)

            with torch.no_grad():
                img_gen = self._g_ema(lbl)

                h, _ = self._encoder(img)
                h_gen, _ = self._encoder(img_gen)

            activations_real.extend(h.cpu().numpy())
            activations_fake.extend(h_gen.cpu().numpy())

        activations_real = np.array(activations_real)
        activations_fake = np.array(activations_fake)

        mu_real = np.mean(activations_real, axis=0)
        sigma_real = np.cov(activations_real, rowvar=False)

        mu_fake = np.mean(activations_fake, axis=0)
        sigma_fake = np.cov(activations_fake, rowvar=False)
        fletcher_distance = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
        return fletcher_distance

    @torch.no_grad()
    def _compute_geometric_distance(self) -> torch.Tensor:

        n_batches = 200

        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")
        loader = self._get_dl()

        # real data embeddings
        real_embeddings = []
        for _ in trange(n_batches):
            img, _ = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            real_embeddings.extend(h.cpu())
        real_embeddings = torch.stack(real_embeddings)

        # fake embeddings
        fake_embeddings = []
        for _ in trange(n_batches):
            label_oh = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label_oh)
                h, _ = self._encoder(img)

            fake_embeddings.extend(h.cpu())
        fake_embeddings = torch.stack(fake_embeddings)
        distance = loss(real_embeddings, fake_embeddings)
        return distance

    @torch.no_grad()
    def _attribute_control_accuracy(self) -> Dict:
        """Computes attribute control accuracy

        Returns:
            Dict: attribute control accuracy for each label
        """

        n_out = self._config['dataset']['n_out']
        bs = self._config['batch_size']
        diffs = []

        for _ in trange(n_out):
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

    @torch.no_grad()
    def _compute_chamfer_distance(self) -> float:
        """Computes Chamfer distance between real and generated data

        Returns:
            float: computed Chamfer distance
        """

        n_batches = 20

        loader = self._get_dl()
        embeddings = []
        # real data embeddings
        for _ in trange(n_batches):
            img, _ = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())

        # generated data embeddings
        for _ in trange(n_batches):
            label = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label)
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

    @torch.no_grad()
    def _explore_eps_zs(self) -> None:
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

    @torch.no_grad()
    def _explore_eps(self) -> None:
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

    @torch.no_grad()
    def _traverse_zk(self) -> None:
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

    @torch.no_grad()
    def _explore_y(self) -> None:
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

    @torch.no_grad()
    def _display_output_eps(self) -> None:
        """Displays TSNE of the epsilon from real dataset and from generated dataset"""

        n_classes = self._config['dataset']['n_out']
        n_batches = 200

        loader = self._get_dl()
        labels, embeddings = [], []

        # real data embeddings
        for _ in trange(n_batches):
            img, label = next(loader)
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            labels.extend(label.cpu().numpy().tolist())
            embeddings.extend(h.cpu().numpy())

        # generated data embeddings
        for _ in trange(n_batches):
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine((0, 360), (0.01,) * 2),
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

    @torch.no_grad()
    def _compute_fid_score(self) -> float:
        """Computes FID score for the dataset

        Returns:
            float: FID score
        """

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 299

        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        ds = make_dl.dataset_test

        n_samples = len(ds)
        fid_func = get_fid_fn(ds, self._device, bs, n_samples)
        with torch.no_grad():
            fid_score = fid_func(self._g_ema)
        return fid_score

    @torch.no_grad()
    def _compute_inception_score(self) -> float:
        """Computes inception score for the dataset

        Returns:
            float: inception score
        """

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, augmented=False)

        n_samples = len(make_dl.dataset_test)
        bs = self._config['batch_size']
        dataset = GANDataset(self._g_ema, n=n_samples)

        with torch.no_grad():
            score = inception_score(dataset, batch_size=bs, resize=True)[0]
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
