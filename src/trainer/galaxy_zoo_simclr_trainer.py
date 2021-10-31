import os
from typing import Dict, NoReturn

import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models import ResNetSimCLR
from src.loss import NTXentLoss
from src.data import GalaxyZooLabeledDataset, GalaxyZooUnlabeledDataset
from src.transform import ContrastiveAugmentor, ValidAugmentor
from src.utils import get_device
from src.utils import SummaryWriterWithSources
from src.utils import tsne_display_tensorboard
from src.utils import PathOrStr


class GalaxyZooSimCLRTrainer:

    """Trainer for training self-supervised SimCLR encoder"""

    def __init__(self, config_path: PathOrStr, config: Dict):
        """
        Args:
            config_path: path to config file
            config: config file
        """

        self._config = config
        self._config_path = config_path
        self._device = get_device()
        self._writer = SummaryWriterWithSources(
            files_to_copy=[config_path],
            experiment_name=self._config['comment']
        )

        self._nt_xent_criterion = NTXentLoss(
            self._device, config["batch_size"], **config["loss"]
        )

    def train(self) -> NoReturn:
        """Trains self-supervised SimCLR encoder"""

        epochs = self._config['epochs']
        log_every = self._config['log_every']
        val_every = self._config['val_every']
        eval_every = self._config['eval_every']
        wd = eval(self._config['wd'])
        lr = eval(self._config['lr'])

        train_loader, valid_loader, test_loader = self._get_dls()
        model = self._load_model()
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader),
                                                         eta_min=0, last_epoch=-1)

        global_step = 0
        best_valid_loss = 0
        for epoch in range(1, epochs + 1):

            for (xis, xjs), c in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
                optimizer.zero_grad()
                loss = self._step(model, xis, xjs)

                if global_step % log_every == 0:
                    self._writer.add_scalar('train/loss', loss, global_step)
                loss.backward()
                optimizer.step()

                global_step += 1

            if epoch == 1 or epoch % val_every == 0:
                valid_loss = self._validate(model, valid_loader)
                self._writer.add_scalar('val/loss', valid_loss, epoch)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self._save_model(model, f'{epoch:04}_best')

            if epoch == 1 or epoch % eval_every == 0:
                self._run_tsne(test_loader, model, epoch)

            self._save_model(model, f'{epoch:04}')

            if epoch >= 10:
                scheduler.step()
            self._writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    def _run_tsne(self, loader: DataLoader, model: nn.Module, epoch: int) -> NoReturn:
        """Runs TSNE dimension reduction and saved plots to tensorboard

        Args:
            loader: dataset loader
            model: encoder, which compute embeddings
            epoch: epoch number
        """

        embeddings_h = []
        embeddings_z = []
        labels = []

        for (img, label) in tqdm(loader):
            img = img.to(self._device)

            with torch.no_grad():
                h, z = model(img)

            h_np = h.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()
            label = label.numpy()

            embeddings_h.extend(h_np)
            embeddings_z.extend(z_np)
            labels.extend(label)

        embeddings_h = np.array(embeddings_h)
        embeddings_z = np.array(embeddings_z)
        labels = np.array(labels)

        # compute tsne
        tsne_h = TSNE(n_components=2).fit_transform(embeddings_h)
        tsne_z = TSNE(n_components=2).fit_transform(embeddings_z)

        titles = ['Smooth, disk or artifact/star',
                  'Edge on disk, not edge on',
                  'Barred, not barred',
                  'Spiral arms, no spiral arms',
                  'Bulge: no bulge, noticeable bulge, obvious bulge, dominant bulge',
                  'Anything odd: odd, not odd',
                  'Roundness: completely round, elliptic, cigar-shaped',
                  'Odd features: ring, lens, disturbed, irregular, other, merger, dust lane',
                  'Bulge shape: rounded, boxy, no bulge',
                  'Tightness of spiral arms: tight, medium, loose',
                  'Number of spiral arms: 1, 2, 3, 4, 5, 5+, can`t tell']

        indices = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 13), (13, 15), (15, 18), (18, 25), (25, 28), (28, 31), (31, 37)]

        for i in range(len(titles)):
            labels_sel = labels[:, indices[i][0]:indices[i][1]]
            labels_sel = np.argmax(labels_sel, axis=1)

            img_tsne_h = tsne_display_tensorboard(tsne_h, title=f'{titles[i]} (H)', c_vector=labels_sel)
            img_tsne_z = tsne_display_tensorboard(tsne_z, title=f'{titles[i]} (Z)', c_vector=labels_sel)
            self._writer.add_image(f'{titles[i]} (H)', img_tsne_h, epoch)
            self._writer.add_image(f'{titles[i]} (Z)', img_tsne_z, epoch)

    def _get_dls(self):
        """Returns train, validation and test dataloader"""

        batch_size = self._config['batch_size']
        input_shape = eval(self._config['input_size'])
        cpu_count = os.cpu_count()

        dataset_path = self._config['dataset']['path']
        contrastive_transform = ContrastiveAugmentor(input_shape, 'galaxy_zoo')
        dataset = GalaxyZooUnlabeledDataset(dataset_path, transform=contrastive_transform)

        n = len(dataset)
        n_train = int(0.95 * n)

        train_dataset = Subset(dataset, indices=range(0, n_train))
        valid_dataset = Subset(dataset, indices=range(n_train, n))
        train_loader = DataLoader(train_dataset, batch_size, num_workers=cpu_count,
                                  shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size, num_workers=cpu_count,
                                  shuffle=True, drop_last=True)

        test_ds_path = self._config['dataset']['test_path']
        test_ds_anno = self._config['dataset']['test_anno']
        test_transform = ValidAugmentor(input_shape, 'galaxy_zoo')
        test_dataset = GalaxyZooLabeledDataset(test_ds_path, test_ds_anno, transform=test_transform)
        n_test = int(0.1 * len(test_dataset))
        test_dataset = Subset(test_dataset, np.random.randint(0, len(test_dataset), n_test))
        test_loader = DataLoader(test_dataset, batch_size, num_workers=cpu_count,
                                 shuffle=True, drop_last=True, pin_memory=True)
        return train_loader, valid_loader, test_loader

    def _step(self,
              model: nn.Module,
              xis: torch.Tensor,
              xjs: torch.Tensor) -> torch.Tensor:
        """SimCLR training step"""

        xis = xis.to(self._device)
        xjs = xjs.to(self._device)

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self._nt_xent_criterion(zis, zjs)

        return loss

    def _load_model(self) -> nn.Module:
        model_path = self._config['fine_tune_from']
        base_model = self._config['model']['base_model']
        n_channels = self._config['model']['n_channels']
        out_dim = self._config['model']['out_dim']

        model = ResNetSimCLR(base_model, n_channels, out_dim)

        try:
            if model_path is not None:
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                print(f'Loaded: {model_path}')
            else:
                print('Training model from scratch')
        except:
            print('Training model from scratch')

        if torch.cuda.device_count() > 1:
            print(f'Use {torch.cuda.device_count()} GPUs')
            model = nn.DataParallel(model)

        model.to(self._device)
        return model

    def _validate(self, model: nn.Module, valid_loader: DataLoader) -> float:
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.
            counter = 0
            for (xis, xjs), c in valid_loader:
                loss = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def _save_model(self, model: nn.Module, suffix: str) -> NoReturn:
        checkpoint_folder = self._writer.checkpoint_folder
        save_file = checkpoint_folder / f'model_{suffix}.pth'

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_file)
        else:
            torch.save(model.state_dict(), save_file)
