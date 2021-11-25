from typing import Dict, NoReturn, Optional
import copy
from functools import partial
import random

from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils

from .generator_trainer import GeneratorTrainer
from src.models import ConditionalGenerator
from src.models import ResNetSimCLR, GalaxyZooClassifier
from src.models import GlobalDiscriminator, NLayerDiscriminator
from src.loss import get_adversarial_losses, get_regularizer
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
        batch_size = self._config['batch_size']
        cls_reg_every = self._config['cls_reg_every']  # classification consistency regularization
        d_reg_every = self._config['d_reg_every']   # discriminator regularization
        log_every = self._config['log_every']
        sample_every = self._config['sample_every']

        train_dl, val_dl = self._get_dl()

        log_sample = next(train_dl)[1][:16]
        log_sample = log_sample.to(self._device)

        samples_folder = self._writer.checkpoint_folder.parent / 'samples'
        samples_folder.mkdir(exist_ok=True, parents=True)

        ema = partial(accumulate, decay=0.5 ** (batch_size / (10 * 1000)))

        step = 0
        for epoch in trange(self._epoch, epochs, desc='Epochs'):
            self._generator.train()
            self._discriminator.train()

            for (real_img, real_label) in tqdm(train_dl, desc=f'Epoch: {epoch}/{epochs}'):
                real_img = real_img.to(self._device)
                real_label = real_label.to(self._device)

                # classification regularization
                if step % cls_reg_every == 0:
                    loss_cls = self._step_cls_reg(real_img, real_label)

                # D update
                loss_d = self._step_d(real_img, real_label)
                # D regularize
                if step % d_reg_every == 0:
                    r1 = self._step_reg_d(real_img, real_label)
                # G update
                loss_g, loss_g_adv, loss_g_reg = self._step_g()
                # run exponential moving average weight update
                ema(self._g_ema, self._generator)

                # log
                if step % log_every == 0:
                    self._writer.add_scalar('loss/cls_loss', loss_cls.item(), step)
                    self._writer.add_scalar("loss/D", loss_d.item(), step)
                    self._writer.add_scalar("loss/D_r1", r1.item(), step)
                    self._writer.add_scalar("loss/G", loss_g.item(), step)
                    self._writer.add_scalar("loss/G_orth", loss_g_reg.item(), step)
                    self._writer.add_scalar("loss/G_adv", loss_g_adv.item(), step)

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

            # TODO: Add validation
            self._save_model(epoch)

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
                label_real: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            label_fake = self._sample_label(real=bool(random.getrandbits(1)), add_noise=True)
            img_fake = self._generator(label_fake)

        pred_real = self._discriminator(self._augment(img_real))
        pred_fake = self._discriminator(self._augment(img_fake))
        d_loss = self._d_adv_loss(pred_real, pred_fake)
        self._discriminator.zero_grad()
        d_loss.backward()
        self._d_optim.step()
        return d_loss

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
        ckpt = torch.load(encoder_path)
        encoder.load_state_dict(ckpt)
        encoder = encoder.to(self._device).eval()

        # linear classifier (pretrained)
        classifier_path = self._config['classifier']['path']
        n_feat = self._config['classifier']['n_features']

        classifier = GalaxyZooClassifier(n_feat, n_classes)
        ckpt = torch.load(classifier_path)
        classifier.load_state_dict(ckpt)
        classifier = classifier.to(self._device).eval()

        # generator
        z_size = self._config['generator']['z_size']  # size of the input noise
        n_basis = self._config['generator']['n_basis']  # size of the z1, ..., z1 vectors
        noise_dim = self._config['generator']['noise_size']  # size of the noise after adapter, which mixes y and z
        y_type = self._config['generator']['y_type']

        generator = ConditionalGenerator(
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

        epoch = 0
        if fine_tune_from is not None:
            ckpt = torch.load(fine_tune_from, map_location="cpu")
            epoch = ckpt["epoch"]

            generator.load_state_dict(ckpt["g"])
            discriminator.load_state_dict(ckpt["d"])
            g_ema.load_state_dict(ckpt["g_ema"])
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
            print(f'Loaded from {fine_tune_from}')

        return epoch, generator, discriminator, g_ema, g_optim, d_optim, encoder, classifier

    def _save_model(self, epoch: int):
        ckpt = {
            'epoch': epoch,
            'config': self._config,
            'g': self._generator.state_dict(),
            'd': self._discriminator.state_dict(),
            'g_ema': self._g_ema.state_dict(),
            'g_optim': self._g_optim.state_dict(),
            'd_optim': self._d_optim.state_dict(),
        }

        compute_fid = self._config['fid']

        if compute_fid:
            # TODO: calculate FID on SSL features
            fid_score = self._compute_fid_score()
            ckpt['fid'] = fid_score
            self._writer.add_scalar('FID', fid_score, epoch)

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
                labels += torch.randn(labels.shape)
        else:
            labels = torch.randn((n, n_out))

        labels = make_galaxy_labels_hierarchical(labels)
        labels = labels.to(self._device)
        return labels