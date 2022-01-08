from typing import Dict, NoReturn, Optional
import copy
from functools import partial
import random

from tqdm import trange, tqdm
import numpy as np
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils
from chamferdist import ChamferDistance

from .generator_trainer import GeneratorTrainer, calculate_frechet_distance
from src.models import ConditionalGenerator
from src.models import ResNetSimCLR, GalaxyZooClassifier
from src.models import GlobalDiscriminator, NLayerDiscriminator
from src.models.fid import load_patched_inception_v3, get_fid_between_datasets
from src.loss import get_adversarial_losses, get_regularizer
from src.data import infinite_loader
from src.data.dataset_updated import MakeDataLoader
from src.transform import image_generation_augment
from src.utils import PathOrStr, accumulate, make_galaxy_labels_hierarchical


torch.backends.cudnn.benchmark = True


class GalaxyZooInfoSCC_Trainer(GeneratorTrainer):

    """Updated trainer for conditional generation of Galaxy Zoo dataset"""

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        super().__init__(config_path, config)

        self._epoch, \
            self._generator, self._discriminator, self._g_ema, \
            self._g_optim, self._d_optim, \
            self._encoder, self._classifier = self._load_model()

        self._d_adv_loss, self._g_adv_loss, self._d_reg_loss, self._cls_loss = self._get_loss()
        self._augment = image_generation_augment()

    def train(self) -> NoReturn:

        epochs = self._config['epochs']
        save_every = self._config['save_every']
        batch_size = self._config['batch_size']
        cls_reg_every = self._config['cls_reg_every']  # classification consistency regularization
        d_reg_every = self._config['d_reg_every']   # discriminator regularization
        log_every = self._config['log_every']
        sample_every = self._config['sample_every']

        train_dl, val_dl = self._get_dl()

        for b in train_dl:
            log_sample = b[1][:16]
            break
        log_sample = log_sample.to(self._device)

        samples_folder = self._writer.checkpoint_folder.parent / 'samples'
        samples_folder.mkdir(exist_ok=True, parents=True)

        ema = partial(accumulate, decay=0.5 ** (batch_size / (10 * 1000)))

        step = 0
        for epoch in trange(self._epoch, epochs + 1, desc='Epochs'):
            self._generator.train()
            self._discriminator.train()

            for (real_img, real_label) in tqdm(train_dl, desc=f'Epoch: {epoch}/{epochs}'):
                real_img = real_img.to(self._device)
                real_label = real_label.to(self._device)

                # classification regularization
                if step % cls_reg_every == 0:
                    loss_cls = self._step_cls_reg(real_img, real_label)

                # D update
                loss_d, pred_real, pred_fake = self._step_d(real_img, real_label)
                # D regularize
                if step % d_reg_every == 0:
                    r1 = self._step_reg_d(real_img, real_label)
                # G update
                loss_g, loss_g_adv, loss_g_reg = self._step_g()
                # run exponential moving average weight update
                ema(self._g_ema, self._generator)

                # log
                if step % log_every == 0:
                    self._writer.add_scalar('train/cls_loss', loss_cls.item(), step)
                    self._writer.add_scalar('train/loss_D', loss_d.item(), step)
                    self._writer.add_scalar('train/r1', r1.item(), step)
                    self._writer.add_scalar('train/loss_G', loss_g.item(), step)
                    self._writer.add_scalar('train/loss_G_orth', loss_g_reg.item(), step)
                    self._writer.add_scalar('train/loss_G_adv', loss_g_adv.item(), step)
                    self._writer.add_scalar('train/D(X)', pred_real.item(), step)
                    self._writer.add_scalar('train/D(G(z))', pred_fake.item(), step)

                if step % sample_every == 0:
                    with torch.no_grad():
                        utils.save_image(
                            self._g_ema(log_sample),
                            samples_folder / f'{step:07}.png',
                            nrow=4,
                            normalize=True,
                            value_range=(-1, 1),
                        )

                step += 1

            self._save_model(epoch, compute_metrics=epoch in [1, epochs] or epoch % save_every == 0)

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
        self._writer.add_scalar('eval/FID', fid_score, 0)

        i_score = self._compute_inception_score()
        self._writer.add_scalar('eval/IS', i_score, 0)

        chamfer_dist = self._chamfer_distance()
        self._writer.add_scalar('eval/Chamfer', float(chamfer_dist), 0)

        ssl_fid = self._compute_ssl_fid()
        self._writer.add_scalar('eval/SSL_FID', ssl_fid, 0)

        ssl_ppl = self._compute_ssl_ppl()
        self._writer.add_scalar('eval/SSL_PPL', ssl_ppl, 0)

        vgg_ppl = self._compute_vgg16_ppl()
        self._writer.add_scalar('eval/VGG_PPL', vgg_ppl, 0)

        kid_inception = self._compute_inception_kid()
        self._writer.add_scalar('eval/KID_Inception', kid_inception, 0)

        kid_ssl = self._compute_ssl_kid()
        self._writer.add_scalar('eval/KID_SSL', kid_ssl, 0)
        self._writer.close()

    def compute_baseline(self) -> NoReturn:
        fid_score = self._baseline_ssl_fid()
        self._writer.add_scalar('baseline/FID', fid_score, 0)

        fid_ssl = self._compute_baseline_ssl_fid()
        self._writer.add_scalar('baseline/SSL_FID', fid_ssl, 0)

        chamfer_dist = self._baseline_chamfer_distance()
        self._writer.add_scalar('baseline/Chamfer', chamfer_dist, 0)

        kid = self._compute_baseline_inception_kid()
        self._writer.add_scalar('baseline/KID', kid, 0)

        kid_ssl = self._compute_baseline_ssl_kid()
        self._writer.add_scalar('baseline/SSL_KID', kid_ssl, 0)
        self._writer.close()

    def _step_g(self):

        orth_reg = self._config['orth_reg']

        label_fake = self._sample_label(real=bool(random.getrandbits(1)), add_noise=True)
        img_fake = self._generator(label_fake)
        pred_fake = self._discriminator(self._augment(img_fake))

        g_loss_adv = self._g_adv_loss(pred_fake)
        g_loss_reg = self._generator.orthogonal_regularizer() * orth_reg
        g_loss = g_loss_adv + g_loss_reg

        self._generator.zero_grad()
        g_loss.backward()
        self._g_optim.step()
        return g_loss, g_loss_adv, g_loss_reg

    def _step_reg_d(self,
                    img_real: torch.Tensor,
                    label_real: torch.Tensor) -> torch.Tensor:
        d_reg = self._config['d_reg']

        img_real.requires_grad = True
        pred_real = self._discriminator(self._augment(img_real))
        r1 = self._d_reg_loss(pred_real, img_real) * d_reg

        self._discriminator.zero_grad()
        r1.backward()
        self._d_optim.step()
        return r1

    def _step_d(self,
                img_real: torch.Tensor,
                label_real: torch.Tensor):
        with torch.no_grad():
            label_fake = self._sample_label(real=bool(random.getrandbits(1)), add_noise=True)
            img_fake = self._generator(label_fake)

        pred_real = self._discriminator(self._augment(img_real))
        pred_fake = self._discriminator(self._augment(img_fake))
        d_loss = self._d_adv_loss(pred_real, pred_fake)
        self._discriminator.zero_grad()
        d_loss.backward()
        self._d_optim.step()
        return d_loss, pred_real.mean(), pred_fake.mean()

    def _step_cls_reg(self,
                      img_real: torch.Tensor,
                      label_real: torch.Tensor) -> torch.Tensor:
        self._generator.zero_grad()
        img_out = self._generator(label_real)
        h_out, _ = self._encoder(img_out)
        pred = self._classifier(h_out)

        loss_cls = self._cls_loss(pred, label_real)
        loss_cls.backward()
        self._g_optim.step()
        return loss_cls

    def _get_dl(self):
        batch_size = self._config['batch_size']
        n_workers = self._config['n_workers']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        train_dl = make_dl.get_data_loader_train(batch_size=batch_size, shuffle=True,
                                                 num_workers=n_workers)
        val_dl = make_dl.get_data_loader_valid(batch_size=batch_size, shuffle=False,
                                               num_workers=n_workers)
        return train_dl, val_dl

    def _load_model(self):

        lr = eval(self._config['lr'])
        img_size = self._config['dataset']['size']  # size of the images (input and generated)
        n_channels = self._config['dataset']['n_channels']  # number of channels in the images (input and generated)
        n_classes = self._config['dataset']['n_out']  # number of classes
        fine_tune_from = self._config['fine_tune_from']

        # load encoder (pretrained)
        encoder_path = self._config['encoder']['path']
        base_model = self._config['encoder']['base_model']
        out_dim = self._config['encoder']['out_dim']

        encoder = ResNetSimCLR(base_model, n_channels, out_dim)
        ckpt = torch.load(encoder_path, map_location='cpu')
        encoder.load_state_dict(ckpt)
        encoder = encoder.to(self._device).eval()

        # linear classifier (pretrained)
        classifier_path = self._config['classifier']['path']
        n_feat = self._config['classifier']['n_features']

        classifier = GalaxyZooClassifier(n_feat, n_classes)
        ckpt = torch.load(classifier_path, map_location='cpu')
        classifier.load_state_dict(ckpt)
        classifier = classifier.to(self._device).eval()

        # generator
        z_size = self._config['generator']['z_size']  # size of the input noise
        n_basis = self._config['generator']['n_basis']  # size of the z1, ..., z1 vectors
        noise_dim = self._config['generator']['noise_size']  # size of the noise after adapter, which mixes y and z
        y_type = self._config['generator']['y_type']

        generator = ConditionalGenerator(
            config=self._config,
            size=img_size,
            y_size=n_classes,
            z_size=z_size,
            out_channels=n_channels,
            n_basis=n_basis,
            noise_dim=noise_dim,
            y_type=y_type
        ).to(self._device).train()
        g_ema = copy.deepcopy(generator).eval()

        # discriminator
        disc_type = self._config['discriminator']['type']

        if disc_type == 'global':
            discriminator = GlobalDiscriminator(n_channels, img_size)
        elif disc_type == 'patch':
            ndf = self._config['discriminator']['ndf']  # number of filters
            n_layers = self._config['discriminator']['n_layers']
            actnorm = self._config['discriminator']['actnorm']

            discriminator = NLayerDiscriminator(n_channels, ndf, n_layers, use_actnorm=actnorm)
        else:
            raise ValueError('Unsupported discriminator')

        discriminator = discriminator.to(self._device).train()

        # optimizers
        g_optim = optim.Adam(
            generator.parameters(),
            lr=lr,
            betas=(0.5, 0.99),
        )

        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.99),
        )

        epoch = 1
        if fine_tune_from is not None:
            ckpt = torch.load(fine_tune_from, map_location="cpu")
            epoch = ckpt['epoch'] if 'epoch' in ckpt else 1

            generator.load_state_dict(ckpt["g"])
            discriminator.load_state_dict(ckpt["d"])
            g_ema.load_state_dict(ckpt["g_ema"])
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
            print(f'Loaded from {fine_tune_from}')

        return epoch, generator, discriminator, g_ema, g_optim, d_optim, encoder, classifier

    def _save_model(self, epoch: int, compute_metrics: bool = False):
        ckpt = {
            'epoch': epoch,
            'config': self._config,
            'g': self._generator.state_dict(),
            'd': self._discriminator.state_dict(),
            'g_ema': self._g_ema.state_dict(),
            'g_optim': self._g_optim.state_dict(),
            'd_optim': self._d_optim.state_dict(),
        }

        compute_fid = self._config['fid'] and compute_metrics

        if compute_fid:

            print('start FID')
            fid_score = self._compute_fid_score()
            ckpt['fid'] = fid_score
            self._writer.add_scalar('eval/FID', fid_score, epoch)
            print('end FID')

            print('start SSL FID')
            ssl_fid = self._compute_ssl_fid()
            ckpt['ssl_fid'] = ssl_fid
            self._writer.add_scalar('eval/SSL_FID', ssl_fid, epoch)
            print('End SSL FID')

            print('start Chamfer dist')
            chamfer_dist = self._chamfer_distance()
            ckpt['chamfer_dist'] = chamfer_dist
            self._writer.add_scalar('eval/Chamfer', chamfer_dist, epoch)
            print('end Chamfer dist')

            print('Start SSL PPL')
            ssl_ppl = self._compute_ssl_ppl()
            ckpt['ssl_ppl'] = ssl_ppl
            self._writer.add_scalar('eval/SSL_PPL', ssl_ppl, epoch)
            print('End SSL PPL')

            print('Start VGG PPL')
            vgg_ppl = self._compute_vgg16_ppl()
            ckpt['vgg_ppl'] = vgg_ppl
            self._writer.add_scalar('eval/VGG_PPL', vgg_ppl, epoch)
            print('end VGG PPL')

            print('start KID')
            kid_inception = self._compute_inception_kid()
            ckpt['KID'] = kid_inception
            self._writer.add_scalar('eval/KID_Inception', kid_inception, epoch)
            print('end KID')

            print('start KID SSL')
            kid_ssl = self._compute_ssl_kid()
            ckpt['KID_SSL'] = kid_ssl
            self._writer.add_scalar('eval/KID_SSL', kid_ssl, epoch)
            print('end KID')

        checkpoint_folder = self._writer.checkpoint_folder
        save_file = checkpoint_folder / f'{epoch:07}.pt'
        torch.save(ckpt, save_file)

    def _get_loss(self):
        """Returns loss functions for GAN based on config

        Returns:
            loss functions
        """

        d_adv_loss, g_adv_loss = get_adversarial_losses(self._config['loss'])
        d_reg_loss = get_regularizer("r1")

        cls_loss = nn.MSELoss()
        return d_adv_loss, g_adv_loss, d_reg_loss, cls_loss

    def _sample_label(self,
                      n: Optional[int] = None,
                      real: bool = True,
                      add_noise: bool = False) -> torch.Tensor:
        """Samples y label for the dataset

        Args:
            n: number of labels to sample

            real: if True, then the label will be sampled from the dataset

            add_noise: if True and real is True, some noise will be added to the real label

        Returns:
            torch.Tensor: sampled random label
        """

        n_out = self._config['dataset']['n_out']
        if n is None:
            batch_size = self._config['batch_size']
            n = batch_size

        if real:

            labels = []
            for _ in range(n):
                idx = random.randrange(len(self._sample_ds))
                _, label = self._sample_ds[idx]
                labels.append(torch.from_numpy(label))

            labels = torch.stack(labels)

            if add_noise:
                labels += torch.randn(labels.shape) * 0.1
        else:
            labels = torch.randn((n, n_out))

        labels = make_galaxy_labels_hierarchical(labels)
        labels = labels.to(self._device)
        return labels

    def _baseline_ssl_fid(self) -> float:
        """Computes baseline FID score

        Returns:
            float: baseline FID score
        """

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test

        fid_score = get_fid_between_datasets(ds_test, ds_val, self._device)
        return fid_score

    def _compute_ssl_fid(self) -> float:
        """Computes FID on SSL features

        Returns:
            float: FID
        """

        bs = self._config['batch_size']

        self._encoder.eval()
        n_batches = int(50_000 / bs) + 1

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = infinite_loader(DataLoader(make_dl.dataset_valid, bs, True))

        # compute stats for real dataset
        real_activations = []
        for i, (img, lbl) in enumerate(tqdm(ds_val, total=n_batches)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            real_activations.extend(h.cpu().numpy())

            if i == n_batches:
                break
        real_activations = np.array(real_activations)
        mu_real = np.mean(real_activations, axis=0)
        sigma_real = np.cov(real_activations, rowvar=False)

        # compute stats for fake dataset
        fake_activations = []
        for _ in trange(n_batches):
            label = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label)
                h, _ = self._encoder(img)
            fake_activations.extend(h.cpu().numpy())
        fake_activations = np.array(fake_activations)

        mu_fake = np.mean(fake_activations, axis=0)
        sigma_fake = np.cov(fake_activations, rowvar=False)

        fletcher_distance = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
        self._encoder.train()
        return fletcher_distance

    def _compute_baseline_ssl_fid(self) -> float:
        """Computes baseline FID on SSL features

        Returns:
            float: baseline SSL FID
        """

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']

        self._encoder.eval()

        n_batches = int(50_000 / bs) + 1
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 299

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))
        dl_test = infinite_loader(DataLoader(ds_test, bs, True, num_workers=n_workers))

        activation_train = []
        for i, (img, _) in enumerate(tqdm(dl_val, total=n_batches)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            activation_train.extend(h.cpu().numpy())

            if i == n_batches:
                break
        activation_train = np.array(activation_train)
        mu_train = np.mean(activation_train, axis=0)
        sigma_train = np.cov(activation_train, rowvar=False)

        # TODO: make the separate function
        activation_test = []
        for i, (img, _) in enumerate(tqdm(dl_test, total=n_batches)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            activation_test.extend(h.cpu().numpy())

            if i == n_batches:
                break
        activation_test = np.array(activation_test)
        mu_test = np.mean(activation_test, axis=0)
        sigma_test = np.cov(activation_test, rowvar=False)
        fletcher_distance = calculate_frechet_distance(mu_train, sigma_train, mu_test, sigma_test)
        return fletcher_distance

    @torch.no_grad()
    def _chamfer_distance(self) -> float:
        """Computes Chamfer distance between real and generated data

        Returns:
            float: computed Chamfer distance
        """

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']

        # train_dl, val_dl = self._get_dl()
        n_batches = int(20_000 / bs) + 1

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))

        embeddings = []
        # real data embeddings
        for i, (img, lbl) in enumerate(tqdm(dl_val)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())

            if i == n_batches:
                break

        # generated data embeddings
        for _ in trange(n_batches):
            label = self._sample_label()

            with torch.no_grad():
                img = self._g_ema(label)
                h, _ = self._encoder(img)

            embeddings.extend(h.cpu().numpy())
        embeddings = np.array(embeddings, dtype=np.float32)
        tsne_emb = TSNE(n_components=3, n_jobs=16).fit_transform(embeddings)
        n = len(tsne_emb)

        tsne_real = np.array(tsne_emb[:n//2, ], dtype=np.float32)
        tsne_fake = np.array(tsne_emb[n//2:, ], dtype=np.float32)

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_fake).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    def _baseline_chamfer_distance(self) -> float:
        """Computed baseline Chamfer distance

        Returns:
            float: baseline Chamfer distance
        """

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']

        n_batches = int(50_000 / bs) + 1
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))
        dl_test = infinite_loader(DataLoader(ds_test, bs, True, num_workers=n_workers))

        embeddings_train = []
        for i, (img, _) in enumerate(tqdm(dl_val, total=n_batches)):
            img = img.to(self._device)
            with torch.no_grad():
                h, _ = self._encoder(img)
            embeddings_train.extend(h.cpu().numpy())

            if i == n_batches:
                break

        embeddings_test = []
        for i, (img, _) in enumerate(tqdm(dl_test, total=n_batches)):
            img = img.to(self._device)
            with torch.no_grad():
                h, _ = self._encoder(img)
            embeddings_test.extend(h.cpu().numpy())

            if i == n_batches:
                break

        tsne_train = TSNE(n_components=3, n_jobs=16).fit_transform(embeddings_train)
        tsne_test = TSNE(n_components=3, n_jobs=16).fit_transform(embeddings_test)

        tsne_real = torch.from_numpy(tsne_train).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_test).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    def _compute_inception_kid(self) -> float:
        """Computes Kernel Inception Distance using features computed using pretrained InceptionV3

        Returns:
            float: KID (lower - better)
        """

        encoder = load_patched_inception_v3().to(self._device).eval()
        self._g_ema.eval()

        # train_dl, val_dl = self._get_dl()
        bs = self._config['batch_size']
        n_workers = self._config['n_workers']

        n_batches = int(50_000 / bs) + 1
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))

        real_features = []
        for i, (img, lbl) in enumerate(tqdm(dl_val, total=n_batches)):
            img = img.to(self._device)

            if img.shape[2] != 299 or img.shape[3] != 299:
                img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

            with torch.no_grad():
                feature = encoder(img)[0].flatten(start_dim=1)
                real_features.extend(feature.cpu().numpy())

            if i == n_batches:
                break

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

    def _compute_baseline_inception_kid(self) -> float:
        """Computes baseline Inception KID

        Returns:
            float: baseline Inception KID
        """

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']
        encoder = load_patched_inception_v3().to(self._device).eval()

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))
        dl_test = infinite_loader(DataLoader(ds_test, bs, True, num_workers=n_workers))

        n_batches = int(50_000 / bs) + 1

        features_train = []
        for i, (img, _) in enumerate(tqdm(dl_val, total=n_batches)):
            img = img.to(self._device)

            if img.shape[2] != 299 or img.shape[3] != 299:
                img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

            with torch.no_grad():
                feature = encoder(img)[0].flatten(start_dim=1)
                features_train.extend(feature.cpu().numpy())

            if i == n_batches:
                break
        features_train = np.array(features_train)

        features_test = []
        for i, (img, _) in enumerate(tqdm(dl_test, total=n_batches)):
            img = img.to(self._device)

            if img.shape[2] != 299 or img.shape[3] != 299:
                img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

            with torch.no_grad():
                feature = encoder(img)[0].flatten(start_dim=1)
                features_test.extend(feature.cpu().numpy())

            if i == n_batches:
                break
        features_test = np.array(features_test)

        m = 1000  # max subset size
        num_subsets = 100

        n = features_train.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = features_train[np.random.choice(features_train.shape[0], m, replace=False)]
            y = features_test[np.random.choice(features_test.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)

    def _compute_ssl_kid(self) -> float:
        """Computes Kernel Inception Distance using features computed using pretrained SimCLR

        Returns:
            float: KID
        """

        self._encoder.eval()
        self._g_ema.eval()

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']
        n_batches = int(50_000 / bs) + 1

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))

        real_features = []
        for i, (img, lbl) in enumerate(tqdm(dl_val, total=n_batches)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            real_features.extend(h.cpu().numpy())

            if i == n_batches:
                break
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

    def _compute_baseline_ssl_kid(self) -> float:
        """Computes baseline Kernel Inception Distance using features computed using pretrained SimCLR

        Returns:
            float: KID
        """

        self._encoder.eval()

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 64

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        dl_val = infinite_loader(DataLoader(ds_val, bs, True, num_workers=n_workers))
        dl_test = infinite_loader(DataLoader(ds_test, bs, True, num_workers=n_workers))

        n_batches = int(50_000 / bs) + 1

        features_train = []
        for i, (img, _) in enumerate(tqdm(dl_val, total=n_batches)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            features_train.extend(h.cpu().numpy())

            if i == n_batches:
                break

        features_train = np.array(features_train)

        features_test = []
        for i, (img, _) in enumerate(tqdm(dl_test, total=n_batches)):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = self._encoder(img)
            features_test.extend(h.cpu().numpy())
            if i == n_batches:
                break

        features_test = np.array(features_test)

        m = 1000  # max subset size
        num_subsets = 100

        n = features_train.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = features_train[np.random.choice(features_train.shape[0], m, replace=False)]
            y = features_test[np.random.choice(features_test.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return kid
