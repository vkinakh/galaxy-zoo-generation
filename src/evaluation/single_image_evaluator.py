from typing import Optional, Dict
from pathlib import Path
from PIL import Image
from pprint import pprint

from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from chamferdist import ChamferDistance
from geomloss import SamplesLoss

from .base_evaluator import BaseEvaluator
from src.models import ResNetSimCLR, ImageClassifier
from src.models.fid import get_fid_fn
from src.models import inception_score
from src.models.fid import load_patched_inception_v3
from src.data.dataset_updated import MakeDataLoader
from src.data import GANDataset
from src.metrics.statistics import get_measures_dataloader, get_measures_generator, evaluate_measures
from src.metrics.distribution_measures.autoencoder import Encoder
from src.metrics.distribution_measures import evaluate_latent_distribution
from src.utils.metrics import calculate_frechet_distance


torch.backends.cudnn.benchmark = True


class SingleImageGenerator:

    """Generator that returns one augmented image

    It is used to simulate the mode collapsed models"""

    def __init__(self, image: Image, size: int, device):
        self._image = image
        self._size = size
        self._device = device
        self._trans = self._get_trans()

        # mock variable
        self.eps_size = 512

    def _get_trans(self):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine((0, 360), (0.01,) * 2),  # rotation, -4 to 4 translation
            transforms.CenterCrop(207),
            transforms.Resize((self._size,) * 2),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        return trans

    def sample(self, bs: int):
        imgs = []
        for _ in range(bs):
            imgs.append(self._trans(self._image))
        return torch.stack(imgs).to(self._device)

    def __call__(self, bs: Optional[int] = None,
                 y: Optional[torch.Tensor] = None,
                 eps: Optional[torch.Tensor] = None,
                 *args, **kwargs):

        if bs is None:
            if y is not None:
                bs = y.shape[0]
            else:
                bs = eps.shape[0]

        return self.sample(bs)


class SingleImageEvaluator(BaseEvaluator):

    def __init__(self, config_path: str):
        super().__init__(config_path)

        self._encoder_simclr, self._autoencoder, self._classifier = self._load_model()

        image = Image.open(self.config['path_image'])
        self._generator = SingleImageGenerator(image, self.config['dataset']['size'], self.device)

        self._columns = ['Class1.1', 'Class1.2', 'Class1.3',  # TODO: get real names of the labels
                         'Class2.1', 'Class2.2',
                         'Class3.1', 'Class3.2',
                         'Class4.1', 'Class4.2',
                         'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4',
                         'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2',
                         'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
                         'Class9.1', 'Class9.2', 'Class9.3',
                         'Class10.1', 'Class10.2', 'Class10.3',
                         'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6']

    def evaluate(self) -> None:
        """Evaluated the InfoSCC-GAN model by computing

        - FID score
        - FID score using SimCLR features
        - FID score using AE features
        - Inception Score
        - Chamfer distance using SimCLR features
        - Chamfer distance using AE features
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
        self._writer.add_scalar('eval/IS', inception_score, 0)

        chamfer_dist = self._compute_chamfer_distance('simclr')
        self._writer.add_scalar('eval/Chamfer_simclr', float(chamfer_dist), 0)

        chamfer_ae = self._compute_chamfer_distance('ae')
        self._writer.add_scalar('eval/Chamfer_ae', float(chamfer_ae), 0)

        kid_inception = self._compute_kid('inception')
        self._writer.add_scalar('eval/KID_Inception', kid_inception, 0)

        kid_ssl = self._compute_kid('simclr')
        self._writer.add_scalar('eval/KID_SSL', kid_ssl, 0)

        kid_ae = self._compute_kid('ae')
        self._writer.add_scalar('eval/KID_AE', kid_ae, 0)

        geometric_dist = self._compute_geometric_distance('simclr')
        self._writer.add_scalar('eval/Geom_dist_simclr', geometric_dist, 0)

        geometric_dist_ae = self._compute_geometric_distance('ae')
        self._writer.add_scalar('eval/Geom_dist_ae', geometric_dist_ae, 0)

        morp_res = self._compute_morphological_features()
        self._log('eval/morphological', morp_res, 0)
        pprint(morp_res)

        attribute_accuracy = self._attribute_control_accuracy()
        self._log('eval/attribute_control_accuracy', attribute_accuracy, 0)
        pprint(attribute_accuracy)

        res_dist_measures = self._compute_distribution_measures()
        print('distribution measures:')
        pprint(res_dist_measures)

        self._writer.close()

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
        dl_test = make_dl.get_data_loader_test(bs)

        # compute activations
        activations_real = []
        activations_fake = []

        for batch_test in dl_test:
            img, _ = batch_test
            img = img.to(self._device)
            bs = img.shape[0]

            with torch.no_grad():
                img_gen = self._generator(bs=bs)
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

        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        ds = make_dl.dataset_test

        n_samples = len(ds)
        fid_func = get_fid_fn(ds, self._device, bs, n_samples)
        with torch.no_grad():
            fid_score = fid_func(self._generator)
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
        dataset = GANDataset(self._generator, n=n_samples)

        with torch.no_grad():
            score = inception_score(dataset, batch_size=bs, resize=True)[0]
        return score

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
            encoder = self._encoder_simclr
        else:
            encoder = self._autoencoder

        bs = self._config['batch_size']

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_test = make_dl.get_data_loader_test(bs)
        embeddings_real = []
        embeddings_gen = []

        for batch_test in dl_test:
            img, _ = batch_test
            bs = img.shape[0]

            img = img.to(self._device)

            with torch.no_grad():
                img_gen = self._generator(bs=bs)
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
            encoder = self._encoder_simclr
        elif encoder_type == 'inception':
            encoder = load_patched_inception_v3().to(self._device).eval()
        else:
            encoder = self._autoencoder

        bs = self._config['batch_size']
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_test = make_dl.get_data_loader_test(bs)

        features_real = []
        features_gen = []

        for batch_test in dl_test:
            img, _ = batch_test
            bs = img.shape[0]

            img = img.to(self._device)

            with torch.no_grad():
                img_gen = self._generator(bs=bs)

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
            encoder = self._encoder_simclr
        else:
            encoder = self._autoencoder

        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")

        bs = self._config['batch_size']

        # load dataset
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_test = make_dl.get_data_loader_test(bs)

        embeddings_real = []
        embeddings_gen = []

        for batch_test in dl_test:
            img, _ = batch_test
            bs = img.shape[0]

            img = img.to(self._device)

            with torch.no_grad():
                img_gen = self._generator(bs)
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

        bs = self._config['batch_size']

        comment = self._config['comment']
        plot_path = Path(f'./images/{comment}/')
        plot_path.mkdir(parents=True, exist_ok=True)

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_test = make_dl.get_data_loader_test(bs)

        target_measures = get_measures_dataloader(dl_test)
        target_measures.clean_measures(clean_morphology)
        generated_measures = get_measures_generator(self._generator, dl_test)
        generated_measures.clean_measures(clean_morphology)
        distances = evaluate_measures(target_measures, generated_measures, plot=True,
                                      name='Single Image',
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

        bs = self.config['batch_size']

        path = self.config['dataset']['path']
        anno = self.config['dataset']['anno']
        size = self.config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_val = make_dl.get_data_loader_valid(bs)

        n_out = self.config['dataset']['n_out']
        diffs = []
        labels = []

        for batch in tqdm(dl_val):
            _, lbl = batch
            lbl = lbl.to(self.device)

            with torch.no_grad():
                img = self._generator(y=lbl)
                img = (img * 0.5) + 0.5
                pred = self._classifier(img)

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

        make_dl = MakeDataLoader(path, anno, size, augmented=False)
        dl_val = make_dl.get_data_loader_valid(batch_size=bs, shuffle=False,
                                               num_workers=n_workers)
        dl_test = make_dl.get_data_loader_test(batch_size=bs, shuffle=False,
                                               num_workers=n_workers)
        # load encoder
        n_cluster = self._config['n_clusters']
        res_clusters, res_wasserstein = evaluate_latent_distribution(self._generator,
                                                                     dl_test, dl_val,
                                                                     self._autoencoder,
                                                                     N_cluster=n_cluster,
                                                                     batch_size=bs)
        res = {**{'cluster': res_clusters}, **{'wassetstein': res_wasserstein}, 'n_cluster': n_cluster}
        return res

    def _load_model(self):

        # load SimCLR encoder
        path_encoder = self.config['encoder']['path']
        base_model = self.config['encoder']['base_model']
        out_dim = self.config['encoder']['out_dim']
        n_channels = self._config['dataset']['n_channels']  # number of channels in the images (input and generated)

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
        classifier.use_label_hierarchy()
        path_classifier = self.config['classifier']['path']
        ckpt = torch.load(path_classifier, map_location='cpu')
        classifier.load_state_dict(ckpt)

        return encoder_simclr, autoencoder, classifier
