from typing import Dict, NoReturn, Optional
import copy
from functools import partial
import random
from pathlib import Path
from pprint import pprint

from tqdm import trange, tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
from chamferdist import ChamferDistance
from geomloss import SamplesLoss

from .generator_trainer import GeneratorTrainer, calculate_frechet_distance
from src.models import ConditionalGenerator
from src.models import ResNetSimCLR, GalaxyZooClassifier
from src.models import GlobalDiscriminator, NLayerDiscriminator
from src.models import ImageClassifier
from src.models.fid import load_patched_inception_v3, get_fid_between_datasets
from src.models.vgg16 import vgg16
from src.loss import get_adversarial_losses, get_regularizer
from src.data.dataset_updated import MakeDataLoader
from src.transform import image_generation_augment
from src.metrics.statistics import get_measures_dataloader, get_measures_generator, evaluate_measures
from src.metrics.distribution_measures import evaluate_latent_distribution, Encoder
from src.utils import PathOrStr, accumulate, make_galaxy_labels_hierarchical
from src.utils.metrics import slerp


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
        """Run training"""

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

    def evaluate(self) -> None:
        """Evaluates the model by computing:

        - FID score
        - Inception Score
        - Chamfer distance
        - FID score using SimCLR features
        - PPL using SimCLR features
        - PPL using VGG features
        - KID score
        - KID score using SimCLR features
        - morphological features
        - Geometric distance
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

        chamfer_dist = self._compute_chamfer_distance('simclr')
        self._writer.add_scalar('eval/Chamfer', float(chamfer_dist), 0)

        chamfer_ae = self._compute_chamfer_distance('ae')
        self._writer.add_scalar('eval/Chamfer_AE', float(chamfer_ae), 0)

        ssl_fid = self._compute_fid('simclr')
        self._writer.add_scalar('eval/SSL_FID', ssl_fid, 0)

        ae_fid = self._compute_fid('ae')
        self._writer.add_scalar('eval/AE_FID', ae_fid, 0)

        ssl_ppl = self._compute_ppl('simclr')
        self._writer.add_scalar('eval/SSL_PPL', ssl_ppl, 0)

        vgg_ppl = self._compute_ppl('vgg')
        self._writer.add_scalar('eval/VGG_PPL', vgg_ppl, 0)

        ae_ppl = self._compute_ppl('ae')
        self._writer.add_scalar('eval/AE_PPL', ae_ppl, 0)

        kid_inception = self._compute_kid('inception')
        self._writer.add_scalar('eval/KID_Inception', kid_inception, 0)

        kid_ssl = self._compute_kid('simclr')
        self._writer.add_scalar('eval/KID_SSL', kid_ssl, 0)

        kid_ae = self._compute_kid('ae')
        self._writer.add_scalar('eval/KID_AE', kid_ae, 0)

        morp_res = self._compute_morphological_features()
        self._log('eval/morphological', morp_res, 0)
        pprint(morp_res)

        geometric_dist = self._compute_geometric_distance('simclr')
        self._writer.add_scalar('eval/Geometric_dist', geometric_dist, 0)

        geometric_dist_ae = self._compute_geometric_distance('ae')
        self._writer.add_scalar('eval/AE_Geometric_dist', geometric_dist_ae, 0)

        attribute_accuracy = self._attribute_control_accuracy()
        self._log('eval/attribute_control_accuracy', attribute_accuracy, 0)
        pprint(attribute_accuracy)

        res_dist_measures = self._compute_distribution_measures()
        print('distribution measures:')
        pprint(res_dist_measures)

        self._traverse_zk()
        self._explore_eps()
        self._explore_eps_zs()

        self._writer.close()

    def compute_baseline(self) -> NoReturn:
        """Computes baseline metrics"""

        fid_score = self._compute_baseline_fid()
        self._writer.add_scalar('baseline/FID', fid_score, 0)

        fid_ssl = self._compute_baseline_ssl_fid('simclr')
        self._writer.add_scalar('baseline/SSL_FID', fid_ssl, 0)

        fid_ae = self._compute_baseline_ssl_fid('ae')
        self._writer.add_scalar('baseline/AE_FID', fid_ae, 0)

        chamfer_dist = self._compute_baseline_chamfer_distance('simclr')
        self._writer.add_scalar('baseline/Chamfer', chamfer_dist, 0)

        chamfer_dist = self._compute_baseline_chamfer_distance('ae')
        self._writer.add_scalar('baseline/AE_Chamfer', chamfer_dist, 0)

        kid = self._compute_baseline_kid('inception')
        self._writer.add_scalar('baseline/KID_Inception', kid, 0)

        kid_ssl = self._compute_baseline_kid('simclr')
        self._writer.add_scalar('baseline/KID_SSL', kid_ssl, 0)

        kid_ae = self._compute_baseline_kid('ae')
        self._writer.add_scalar('baseline/KID_AE', kid_ae, 0)

        geom_dist = self._baseline_geometric_distance()
        self._writer.add_scalar('baseline/Geometric_dist', geom_dist, 0)

        geom_dist_ae = self._baseline_geometric_distance('ae')
        self._writer.add_scalar('baseline/AE_Geometric_dist', geom_dist_ae, 0)
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

            fid_score = self._compute_fid_score()
            ckpt['fid'] = fid_score
            self._writer.add_scalar('eval/FID', fid_score, epoch)

            ssl_fid = self._compute_fid('simclr')
            ckpt['ssl_fid'] = ssl_fid
            self._writer.add_scalar('eval/SSL_FID', ssl_fid, epoch)

            chamfer_dist = self._compute_chamfer_distance()
            ckpt['chamfer_dist'] = chamfer_dist
            self._writer.add_scalar('eval/Chamfer', chamfer_dist, epoch)

            ssl_ppl = self._compute_ppl('simclr')
            ckpt['ssl_ppl'] = ssl_ppl
            self._writer.add_scalar('eval/SSL_PPL', ssl_ppl, epoch)

            vgg_ppl = self._compute_ppl('vgg')
            ckpt['vgg_ppl'] = vgg_ppl
            self._writer.add_scalar('eval/VGG_PPL', vgg_ppl, epoch)

            kid_inception = self._compute_kid('inception')
            ckpt['KID'] = kid_inception
            self._writer.add_scalar('eval/KID_Inception', kid_inception, epoch)

            kid_ssl = self._compute_kid('simclr')
            ckpt['KID_SSL'] = kid_ssl
            self._writer.add_scalar('eval/KID_SSL', kid_ssl, epoch)

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

    @torch.no_grad()
    def _compute_distribution_measures(self) -> Dict[str, float]:
        """Computes distribution measures: cluster measures, Wasserstein measures

        Returns:
            Dict[str, float]: distribution measures
        """

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        n_workers = self._config['n_workers']
        bs = self._config['batch_size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_val = make_dl.get_data_loader_valid(batch_size=bs, shuffle=False,
                                               num_workers=n_workers)
        dl_test = make_dl.get_data_loader_test(batch_size=bs, shuffle=False,
                                               num_workers=n_workers)
        # load encoder
        encoder = Encoder().to(self._device).eval()
        ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
        encoder.load_state_dict(ckpt_encoder)

        n_cluster = self._config['eval']['n_clusters']
        res_clusters, res_wasserstein = evaluate_latent_distribution(self._g_ema,
                                                                     dl_test, dl_val, encoder,
                                                                     N_cluster=n_cluster,
                                                                     batch_size=bs)
        res = {**res_clusters, **res_wasserstein, 'n_cluster': n_cluster}
        return res

    @torch.no_grad()
    def _compute_fid(self, encoder_type: str = 'simclr') -> float:
        """Computes FID on custom features

        Args:
            encoder_type: type of encoder to use. Choices: simclr, ae

        Returns:
            float: FID
        """

        if encoder_type not in ['simclr', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_valid = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        # compute activations
        activations_real = []
        activations_fake = []

        for batch_val, batch_test in zip(dl_valid, dl_test):
            img, _ = batch_test
            _, lbl = batch_val
            img = img.to(self._device)
            lbl = lbl.to(self._device)

            with torch.no_grad():
                img_gen = self._g_ema(lbl)

                h, _ = encoder(img)
                h_gen, _ = encoder(img_gen)

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
    def _compute_chamfer_distance(self, encoder_type: str = 'simclr') -> float:
        """Computes Chamfer distance between real and generated data

        Args:
            encoder_type: type of encoder to use. Choices: simclr, ae

        Returns:
            float: computed Chamfer distance
        """

        if encoder_type not in ['simclr', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_valid = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)
        embeddings_real = []
        embeddings_gen = []

        for batch_val, batch_test in zip(dl_valid, dl_test):
            img, _ = batch_test
            _, lbl = batch_val

            img = img.to(self._device)
            lbl = lbl.to(self._device)

            with torch.no_grad():
                img_gen = self._g_ema(lbl)
                h, _ = encoder(img)
                h_gen, _ = encoder(img_gen)

            embeddings_real.extend(h.cpu().numpy())
            embeddings_gen.extend(h_gen.cpu().numpy())

        embeddings_real = np.array(embeddings_real, dtype=np.float32)
        embeddings_gen = np.array(embeddings_gen, dtype=np.float32)
        embeddings = np.concatenate((embeddings_real, embeddings_gen))
        tsne_emb = TSNE(n_components=3, n_jobs=16).fit_transform(embeddings)

        n = len(tsne_emb)
        tsne_real = np.array(tsne_emb[:n//2, ], dtype=np.float32)
        tsne_fake = np.array(tsne_emb[n//2:, ], dtype=np.float32)

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_fake).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    @torch.no_grad()
    def _compute_ppl(self, encoder_type: str = 'simclr') -> float:
        """Computes perceptual path length (PPL)

        Args:
            encoder_type: type of encoder to use. Choices: simclr, vgg, ae

        Returns:
            float: perceptual path length (smaller better)
        """

        if encoder_type not in ['simclr', 'vgg', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        elif encoder_type == 'vgg':
            encoder = vgg16().to(self._device).eval()
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        n_samples = 50_000
        eps = 1e-4
        bs = self._config['batch_size']
        n_batches = int(n_samples / bs) + 1

        dist = []
        for _ in trange(n_batches):
            label = self._sample_label(real=True, add_noise=False)

            labels_cat = torch.cat([label, label])
            t = torch.rand([label.shape[0]], device=label.device)
            eps0 = self._g_ema.sample_eps(bs)
            eps1 = self._g_ema.sample_eps(bs)

            z0 = z1 = self._g_ema.sample_zs(bs)

            epst0 = slerp(eps0, eps1, t.unsqueeze(1))
            epst1 = slerp(eps0, eps1, t.unsqueeze(1) + eps)

            with torch.no_grad():
                img = self._g_ema(labels_cat, torch.cat([epst0, epst1]), torch.cat([z0, z1]))

                if encoder_type in ['simclr', 'ae']:
                    h, _ = encoder(img)
                else:
                    if img.shape[2] != 256 or img.shape[3] != 256:
                        img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bicubic')
                    h = encoder(img)

            h0, h1 = h.chunk(2)
            d = (h0 - h1).square().sum(1) / eps ** 2
            dist.extend(d.cpu().numpy())

        dist = np.array(dist)
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return ppl

    @torch.no_grad()
    def _compute_kid(self, encoder_type: str = 'simclr') -> float:
        """Computes KID score

        Args:
            encoder_type: type of encoder to use. Choices: simclr, inception, ae

        Returns:
            float: KID score
        """

        if encoder_type not in ['simclr', 'inception', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        elif encoder_type == 'inception':
            encoder = load_patched_inception_v3().to(self._device).eval()
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        bs = self._config['batch_size']
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_valid = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        features_real = []
        features_gen = []

        for batch_val, batch_test in zip(dl_valid, dl_test):
            _, lbl = batch_val
            img, _ = batch_test

            img = img.to(self._device)
            lbl = lbl.to(self._device)

            with torch.no_grad():
                img_gen = self._g_ema(lbl)

                if encoder_type == 'inception':
                    if img.shape[2] != 299 or img.shape[3] != 299:
                        img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

                    img_gen = torch.nn.functional.interpolate(img_gen, size=(299, 299), mode='bicubic')

                    h = encoder(img)[0].flatten(start_dim=1)
                    h_gen = encoder(img_gen)[0].flatten(start_dim=1)
                else:
                    h, _ = encoder(img)
                    h_gen, _ = encoder(img_gen)

            features_real.extend(h.cpu().numpy())
            features_gen.extend(h_gen.cpu().numpy())

        features_real = np.array(features_real)
        features_gen = np.array(features_gen)
        m = 1000  # max subset size
        num_subsets = 100

        n = features_real.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = features_gen[np.random.choice(features_gen.shape[0], m, replace=False)]
            y = features_real[np.random.choice(features_real.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def _compute_geometric_distance(self, encoder_type: str = 'simclr') -> float:
        """Computes geometric distance between real and generated samples using
        features computed using SimCLR

        Args:
            encoder_type: type of encoder to use. Choices: simclr, ae

        Returns:
              float: geometric distance
        """

        if encoder_type not in ['simclr', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

        bs = self._config['batch_size']

        # load dataset
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_valid = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        embeddings_real = []
        embeddings_gen = []

        for batch_val, batch_test in zip(dl_valid, dl_test):
            _, lbl = batch_val
            img, _ = batch_test

            img = img.to(self._device)
            lbl = lbl.to(self._device)

            with torch.no_grad():
                img_gen = self._g_ema(lbl)
                h, _ = encoder(img)
                h_gen, _ = encoder(img_gen)

            embeddings_real.extend(h.detach().cpu())
            embeddings_gen.extend(h_gen.detach().cpu())

        embeddings_real = torch.stack(embeddings_real)
        embeddings_gen = torch.stack(embeddings_gen)
        distance = loss(embeddings_real, embeddings_gen)
        return distance.detach().cpu().item()

    @torch.no_grad()
    def _compute_morphological_features(self) -> Dict[str, float]:
        clean_morphology = {
            "deviation": (0, 999999),
            "asymmetry": (-1, 1),
            "smoothness": (-0.5, 1)
        }

        self._g_ema.eval()
        bs = self._config['batch_size']

        comment = self._config['comment']
        plot_path = Path(f'./images/{comment}/')
        plot_path.mkdir(parents=True, exist_ok=True)

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_test = make_dl.get_data_loader_test(bs)
        dl_val = make_dl.get_data_loader_valid(bs)

        target_measures = get_measures_dataloader(dl_test)
        target_measures.clean_measures(clean_morphology)
        generated_measures = get_measures_generator(self._g_ema, dl_val)
        generated_measures.clean_measures(clean_morphology)
        distances = evaluate_measures(target_measures, generated_measures, plot=True,
                                      name='InfoSCC-GAN',
                                      plot_path=plot_path)
        return distances

    @torch.no_grad()
    def _attribute_control_accuracy(self, build_hist: bool = True) -> Dict:
        """Computes attribute control accuracy

        Args:
            build_hist: if True, the histogram of differences for each label will be built and saved

        Returns:
            Dict: attribute control accuracy for each label
        """

        # load classifier
        cls = ImageClassifier().to(self._device).eval()
        cls.use_label_hierarchy()

        path_cls = self._config['eval']['path_classifier']
        ckpt = torch.load(path_cls)
        cls.load_state_dict(ckpt)

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_val = make_dl.get_data_loader_valid(bs)

        n_out = self._config['dataset']['n_out']
        diffs = []
        labels = []

        for batch in tqdm(dl_val):
            _, lbl = batch
            lbl = lbl.to(self._device)

            with torch.no_grad():
                img = self._g_ema(lbl)
                img = (img * 0.5) + 0.5
                pred = cls(img)

            diff = (lbl - pred) ** 2
            diffs.extend(diff.detach().cpu().numpy())
            labels.extend(lbl.detach().cpu().numpy())

        diffs = np.array(diffs)
        labels = np.array(labels)

        if build_hist:
            save_dir = self._writer.checkpoint_folder.parent / 'attribute_control_accuracy'
            save_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_out):
                column = self._columns[i]
                plt.figure()
                plt.title(f'{column}. Attribute control accuracy')
                plt.hist(diffs[:, i], bins=100)
                plt.savefig(save_dir / f'{column}.png', dpi=300)

        mean_diffs = np.mean(diffs, axis=0)

        result = {}
        for i in range(n_out):
            result[self._columns[i]] = mean_diffs[i]

        result['aggregated_attribute_accuracy'] = np.sum(diffs) / np.sum(labels)
        return result

    @torch.no_grad()
    def _baseline_geometric_distance(self, encoder_type: str = 'simclr') -> float:
        """Computes baseline geometric distance between features computed using SimCLR
        for two dataset splits

        Args:
            encoder_type: type of encoder to use. Choices: simclr, ae

        Returns:
            float: baseline geometric distance
        """

        if encoder_type not in ['simclr', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

        bs = self._config['batch_size']
        # load dataset
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_val = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        embeddings_val = []
        embeddings_test = []

        for batch_val, batch_test in zip(dl_val, dl_test):
            img_val, _ = batch_val
            img_val = img_val.to(self._device)
            img_test, _ = batch_test
            img_test = img_test.to(self._device)

            with torch.no_grad():
                h_val, _ = encoder(img_val)
                h_test, _ = encoder(img_test)

            embeddings_val.extend(h_val.detach().cpu())
            embeddings_test.extend(h_test.detach().cpu())

        embeddings_val = torch.stack(embeddings_val)
        embeddings_test = torch.stack(embeddings_test)
        distance = loss(embeddings_val, embeddings_test)
        return distance.detach().cpu().item()

    def _compute_baseline_fid(self) -> float:
        """Computes baseline FID score

        Returns:
            float: baseline FID score
        """

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test

        num_samples = min(len(ds_val), len(ds_test))

        fid_score = get_fid_between_datasets(ds_test, ds_val, self._device, bs, num_samples)
        return fid_score

    @torch.no_grad()
    def _compute_baseline_ssl_fid(self, encoder_type: str = 'simclr') -> float:
        """Computes baseline FID on features, computed using SimCLR and AE

        Args:
            encoder_type: type of encoder to use. Choices: simclr, ae

        Returns:
            float: baseline SSL FID
        """

        if encoder_type not in ['simclr', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_val = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        activations_val = []
        activations_test = []

        for batch_val, batch_test in zip(dl_val, dl_test):
            img_val, _ = batch_val
            img_val = img_val.to(self._device)
            img_test, _ = batch_test
            img_test = img_test.to(self._device)

            with torch.no_grad():
                h_val, _ = encoder(img_val)
                h_test, _ = encoder(img_test)

            activations_val.extend(h_val.cpu().numpy())
            activations_test.extend(h_test.cpu().numpy())

        activations_val = np.array(activations_val)
        mu_val = np.mean(activations_val, axis=0)
        sigma_val = np.cov(activations_val, rowvar=False)

        activations_test = np.array(activations_test)
        mu_test = np.mean(activations_test, axis=0)
        sigma_test = np.cov(activations_test, rowvar=False)
        fletcher_distance = calculate_frechet_distance(mu_val, sigma_val, mu_test, sigma_test)
        return fletcher_distance

    @torch.no_grad()
    def _compute_baseline_chamfer_distance(self, encoder_type: str = 'simclr') -> float:
        """Computed baseline Chamfer distance

        Args:
            encoder_type: type of encoder to use. Choices: simclr, ae

        Returns:
            float: baseline Chamfer distance
        """

        if encoder_type not in ['simclr', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        bs = self._config['batch_size']
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_val = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        embeddings_val = []
        embeddings_test = []
        for batch_val, batch_test in zip(dl_val, dl_test):
            img_val, _ = batch_val
            img_test, _ = batch_test
            img_val = img_val.to(self._device)
            img_test = img_test.to(self._device)

            with torch.no_grad():
                h_val, _ = encoder(img_val)
                h_test, _ = encoder(img_test)

            embeddings_val.extend(h_val.cpu().numpy())
            embeddings_test.extend(h_test.cpu().numpy())

        embeddings_val = np.array(embeddings_val, dtype=np.float32)
        embeddings_test = np.array(embeddings_test, dtype=np.float32)
        embeddings = np.concatenate((embeddings_val, embeddings_test))
        tsne_emb = TSNE(n_components=3, n_jobs=16).fit_transform(embeddings)

        n = len(tsne_emb)
        tsne_val = np.array(tsne_emb[:n//2, ], dtype=np.float32)
        tsne_test = np.array(tsne_emb[n//2:, ], dtype=np.float32)

        tsne_val = torch.from_numpy(tsne_val).unsqueeze(0)
        tsne_test = torch.from_numpy(tsne_test).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_val, tsne_test).detach().item()

    @torch.no_grad()
    def _compute_baseline_kid(self, encoder_type: str = 'simclr') -> float:
        """Computes baseline KID

        Args:
            encoder_type: type of encoder to use. Choices: simclr, inception, ae

        Returns:
            float: KID score
        """

        if encoder_type not in ['simclr', 'inception', 'ae']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        elif encoder_type == 'inception':
            encoder = load_patched_inception_v3().to(self._device).eval()
        else:
            encoder = Encoder().to(self._device).eval()
            ckpt_encoder = torch.load(self._config['eval']['path_encoder'])
            encoder.load_state_dict(ckpt_encoder)

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
        dl_val = make_dl.get_data_loader_valid(bs)
        dl_test = make_dl.get_data_loader_test(bs)

        features_val = []
        features_test = []

        for batch_val, batch_test in zip(dl_val, dl_test):
            img_val, _ = batch_val
            img_val = img_val.to(self._device)
            img_test, _ = batch_test
            img_test = img_test.to(self._device)

            with torch.no_grad():
                if encoder_type == 'inception':
                    img_val = torch.nn.functional.interpolate(img_val, size=(299, 299), mode='bicubic')
                    img_test = torch.nn.functional.interpolate(img_test, size=(299, 299), mode='bicubic')

                    h_val = encoder(img_val)[0].flatten(start_dim=1)
                    h_test = encoder(img_test)[0].flatten(start_dim=1)
                else:
                    h_val, _ = encoder(img_val)
                    h_test, _ = encoder(img_test)

            features_val.extend(h_val.cpu().numpy())
            features_test.extend(h_test.cpu().numpy())

        features_val = np.array(features_val)
        features_test = np.array(features_test)
        m = 1000  # max subset size
        num_subsets = 100

        n = features_val.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = features_val[np.random.choice(features_val.shape[0], m, replace=False)]
            y = features_test[np.random.choice(features_test.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)
