from typing import Dict, NoReturn

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import transforms

from src.models import GalaxyZooClassifier
from src.models import ResNetSimCLR
from src.data import GalaxyZooLabeledDataset
from .base_trainer import BaseTrainer


class GalaxyZooClassifierTrainer(BaseTrainer):

    """Trainer for training GalaxyZoo classifier"""

    def __init__(self,
                 config_path: str,
                 config: Dict):
        """

        Args:
            config_path: path to config file
            config: config
        """

        super().__init__(config_path, config)

        self._model, self._optimizer = self._load_model()
        self._criterion = nn.MSELoss()

    def train(self) -> NoReturn:

        epochs = self._config['epochs']

        train_dl, test_dl = self._get_dls()

        for epoch in range(1, epochs + 1):
            total_loss = 0

            for emd, label in tqdm(train_dl, desc=f'Epoch: {epoch}/{epochs}'):
                emd, label = emd.to(self._device), label.to(self._device)

                self._optimizer.zero_grad()
                logits = self._model(emd)
                loss = self._criterion(logits, label)

                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()

            self._writer.add_scalar('train/loss', total_loss, epoch)
        val_loss = self._eval(test_dl)
        self._writer.add_scalar('val/loss', val_loss, epochs)
        self._save_model('final')

    def _eval(self, loader: DataLoader):
        self._model.eval()

        count = 0
        total_loss = 0

        for (emd, label) in loader:
            emd, label = emd.to(self._device), label.to(self._device)

            with torch.no_grad():
                logits = self._model(emd)

            loss = self._criterion(logits, label)
            count += emd.size(0)
            total_loss += loss.item()

        self._model.train()
        return total_loss / count

    def _get_dls(self):

        batch_size = self._config['batch_size']
        size = self._config['dataset']['size']
        path = self._config['dataset']['path']
        anno_path = self._config['dataset']['anno']
        columns = None if 'columns' not in self._config['dataset'] else self._config['dataset']['columns']

        transform = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Resize((size, size)),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(0.5, 0.5)
        ])

        dataset = GalaxyZooLabeledDataset(path, anno_path, columns, transform)
        n = len(dataset)
        test_ratio = 0.05
        n_train = int(n * (1 - test_ratio))
        train_idx = range(0, n_train)
        test_idx = range(n_train, n)

        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)

        train_emb_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, num_workers=8)
        test_emd_dl = DataLoader(test_ds, batch_size=batch_size, drop_last=True, num_workers=8)

        # load encoder
        encoder_path = self._config['encoder']['path']
        base_model = self._config['encoder']['base_model']
        n_channels = self._config['encoder']['n_channels']
        out_dim = self._config['encoder']['out_dim']
        encoder = ResNetSimCLR(base_model, n_channels, out_dim).to(self._device)
        ckpt = torch.load(encoder_path)
        encoder.load_state_dict(ckpt)
        encoder.eval()

        # compute embeddings
        train_emb = []
        train_labels = []
        test_emb = []
        test_labels = []

        # compute train ds
        for (img, label) in tqdm(train_emb_dl):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = encoder(img)

            embedding = h.detach().cpu().numpy()
            label = label.numpy()

            train_emb.extend(embedding)
            train_labels.extend(label)

        # compute test ds
        for (img, label) in tqdm(test_emd_dl):
            img = img.to(self._device)

            with torch.no_grad():
                h, _ = encoder(img)

            embedding = h.detach().cpu().numpy()
            label = label.numpy()

            test_emb.extend(embedding)
            test_labels.extend(label)

        train_emb = np.array(train_emb, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)

        test_emb = np.array(test_emb, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)

        train_emb_ds = TensorDataset(torch.from_numpy(train_emb), torch.from_numpy(train_labels))
        train_emb_dl = DataLoader(train_emb_ds, batch_size=batch_size, num_workers=8)

        test_emb_ds = TensorDataset(torch.from_numpy(test_emb), torch.from_numpy(test_labels))
        test_emb_dl = DataLoader(test_emb_ds, batch_size=batch_size, num_workers=8)
        return train_emb_dl, test_emb_dl

    def _load_model(self):

        n_features = self._config['model']['n_features']
        n_out = self._config['model']['n_out']
        model_path = self._config['fine_tune_from']
        lr = eval(self._config['lr'])
        wd = eval(self._config['wd'])

        model = GalaxyZooClassifier(n_features, n_out).to(self._device)
        if model_path is not None:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            print(f'Loaded model from: {model_path}')

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
        return model, optimizer

    def _save_model(self, suffix: str) -> NoReturn:
        checkpoint_folder = self._writer.checkpoint_folder
        save_file = checkpoint_folder / f'model_{suffix}.pth'
        torch.save(self._model.state_dict(), save_file)
