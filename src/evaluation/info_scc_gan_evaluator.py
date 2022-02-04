import torch

import numpy as np

from .base_evaluator import BaseEvaluator
from src.models import ConditionalGenerator
from src.models import ResNetSimCLR
from src.models import ImageClassifier
from src.models.fid import get_fid_fn
from src.models import inception_score
from src.data.dataset_updated import MakeDataLoader
from src.data import GANDataset
from src.metrics.distribution_measures.autoencoder import Encoder
from src.utils.metrics import calculate_frechet_distance


torch.backends.cudnn.benchmark = True


class InfoSCCGANEvaluator(BaseEvaluator):

    def __iter__(self, config_path: str):
        super().__init__(config_path)

        self._g_ema, self._encoder_simclr, self._autoencoder, self._classifier = self._load_model()

    def evaluate(self) -> None:
        """Evaluated the InfoSCC-GAN model by computing

        - FID score
        - FID score using SimCLR features
        - FID score using AE features
        - Inception Score
        - Chamfer distance using SimCLR features
        - Chamfer distance using AE features
        - PPL using SimCLR features
        - PPL using VGG features
        - KID score
        - KID score using SimCLR features
        - KID score using AE features
        - morphological features
        - Geometric distance using SimCLR features
        - attribute control accuracy

        """

        fid_inception = self._compute_fid('inception')
        self._writer.add_scalar('eval/FID_inception', fid_inception, 0)

        fid_simclr = self._compute_fid('simclr')
        self._writer.add_scalar('eval/FID_simclr', fid_simclr, 0)

        fid_ae = self._compute_fid('ae')
        self._writer.add_scalar('eval/FID_ae', fid_ae, 0)

        inception_score = self._compute_inception_score()
        self._writer.add_scalar('IS', inception_score, 0)



    @torch.no_grad()
    def _compute_fid(self, encoder_type: str) -> float:
        """Compute FID score

        Args:
            encoder_type: type of encoder to use. Choices: inception, simclr, ae

        Returns:
            float: FID score

        Raises:
            ValueError: if `encoder_type` is incorrect
        """

        if encoder_type not in ['inception', 'simclr', 'ae']:
            raise ValueError('Incorrect encoder type')

        if encoder_type == 'inception':
            return self._compute_fid_inception()

        if encoder_type == 'simclr':
            encoder = self._encoder_simclr
        else:
            encoder = self._autoencoder

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
    def _compute_fid_inception(self) -> float:
        """Computes FID score for the dataset using Inception features

        Returns:
            float: FID score
        """

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = 299

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=False)
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

    def _load_model(self):
        # load generator
        img_size = self._config['dataset']['size']  # size of the images (input and generated)
        n_classes = self._config['dataset']['n_out']  # number of classes
        n_channels = self._config['dataset']['n_channels']  # number of channels in the images (input and generated)

        z_size = self.config['generator']['z_size']  # size of the input noise
        n_basis = self.config['generator']['n_basis']  # size of the z1, ..., z1 vectors
        noise_dim = self.config['generator']['noise_size']  # size of the noise after adapter, which mixes y and z
        y_type = self.config['generator']['y_type']

        g_ema = ConditionalGenerator(
            config=self._config,
            size=img_size,
            y_size=n_classes,
            z_size=z_size,
            out_channels=n_channels,
            n_basis=n_basis,
            noise_dim=noise_dim,
            y_type=y_type
        ).to(self.device).eval()

        path_generator = self.config['generator']['path']
        ckpt = torch.load(path_generator, map_location='cpu')
        g_ema.load_state_dict(ckpt['g_ema'])

        # load SimCLR encoder
        path_encoder = self.config['encoder']['path']
        base_model = self.config['encoder']['base_model']
        out_dim = self.config['encoder']['out_dim']

        encoder_simclr = ResNetSimCLR(base_model, n_channels, out_dim).to(self.device).eval()
        ckpt = torch.load(path_encoder, map_location='cpu')
        encoder_simclr.load_state_dict(ckpt)

        # load autoencoder
        autoencoder = Encoder().to(self.device).eval()
        path_autoencoder = self.config['autoencoder']['path']
        ckpt = torch.load(path_autoencoder, map_location='cpu')
        autoencoder.load_state_dict(ckpt)

        # load classifier
        classifier = ImageClassifier().to(self.device).eval()
        path_classifier = self.config['classifier']['path']
        ckpt = torch.load(path_classifier, map_location='cpu')
        classifier.load_state_dict(ckpt)

        return g_ema, encoder_simclr, autoencoder, classifier
