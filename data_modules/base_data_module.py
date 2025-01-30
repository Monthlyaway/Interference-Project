import lightning as L
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import numpy as np


def load_mat_data(file_path, data_key_contains='Data'):
    """Load data from a .mat file."""
    data = scipy.io.loadmat(file_path)
    for key in data:
        if data_key_contains in key:
            return data[key]
    raise ValueError(f"Data variable containing '{
                     data_key_contains}' not found in {file_path}.")


def load_mat_labels(file_path, label_key_contains='Labels'):
    """Load labels from a .mat file."""
    data = scipy.io.loadmat(file_path)
    for key in data:
        if label_key_contains in key:
            return data[key].flatten()
    raise ValueError(f"Label variable containing '{
                     label_key_contains}' not found in {file_path}.")


class TimeFrequencyDataset(Dataset):
    """Custom Dataset for time and frequency domain data."""

    def __init__(self, signal, spectrum, labels=None):
        super().__init__()
        signal = signal[:, None, :]
        spectrum = spectrum[:, None, :]
        self.signal = torch.tensor(signal)
        self.spectrum = torch.tensor(spectrum)
        self.labels = torch.tensor(labels) if labels is not None else None

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx):
        data = {
            'signal': self.signal[idx],
            'spectrum': self.spectrum[idx]
        }
        if self.labels is not None:
            data['label'] = self.labels[idx]
        return data


class BaseDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

        self.signal_train_path = 'data_modules/1_Signal_Datasets/Signal_TrainingData.mat'
        self.signal_val_path = 'data_modules/1_Signal_Datasets/Signal_ValidationData.mat'
        self.signal_test_path = 'data_modules/1_Signal_Datasets/Signal_TestData.mat'
        self.spectrum_train_path = 'data_modules/2_Spectrm_Datasets/Spectrum_TrainingData.mat'
        self.spectrum_val_path = 'data_modules/2_Spectrm_Datasets/Spectrum_ValidationData.mat'
        self.spectrum_test_path = 'data_modules/2_Spectrm_Datasets/Spectrum_TestData.mat'
        self.signal_test_labels_path = 'data_modules/1_Signal_Datasets/Signal_TestLabels.mat'
        self.spectrum_test_labels_path = 'data_modules/2_Spectrm_Datasets/Spectrum_TestLabels.mat'

    def prepare_data(self):
        signal_train = load_mat_data(self.signal_train_path)
        spectrum_train = load_mat_data(self.spectrum_train_path)

        self.signal_min = signal_train.min()
        self.signal_max = signal_train.max()
        self.spectrum_min = spectrum_train.min()
        self.spectrum_max = spectrum_train.max()

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            signal_train = load_mat_data(self.signal_train_path)
            spectrum_train = load_mat_data(self.spectrum_train_path)
            signal_train = (signal_train - self.signal_min) / \
                (self.signal_max - self.signal_min)
            spectrum_train = (spectrum_train - self.spectrum_min) / \
                (self.spectrum_max - self.spectrum_min)
            self.train_dataset = TimeFrequencyDataset(
                signal_train, spectrum_train)

            signal_val = load_mat_data(self.signal_val_path)
            spectrum_val = load_mat_data(self.spectrum_val_path)
            signal_val = (signal_val - self.signal_min) / \
                (self.signal_max - self.signal_min)
            spectrum_val = (spectrum_val - self.spectrum_min) / \
                (self.spectrum_max - self.spectrum_min)
            self.val_dataset = TimeFrequencyDataset(
                signal_val, spectrum_val)

        if stage == 'test' or stage is None:
            # Load test data
            signal_test = load_mat_data(self.signal_test_path)
            spectrum_test = load_mat_data(self.spectrum_test_path)

            signal_test = (signal_test - self.signal_min) / \
                (self.signal_max - self.signal_min)
            spectrum_test = (spectrum_test - self.spectrum_min) / \
                (self.spectrum_max - self.spectrum_min)

            # Load test labels
            signal_test_labels = load_mat_labels(self.signal_test_labels_path)
            spectrum_test_labels = load_mat_labels(
                self.spectrum_test_labels_path)

            # Create test dataset
            self.test_dataset = TimeFrequencyDataset(
                signal_test, spectrum_test, spectrum_test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)


if __name__ == '__main__':
    dm = BaseDataModule(batch_size=32)
    dm.prepare_data()
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    print(next(iter(train_loader))['signal'])
    print(next(iter(train_loader))['spectrum'])

    dm.setup('test')
    test_loader = dm.test_dataloader()
    print(next(iter(test_loader))['label'])
    print(next(iter(test_loader))['spectrum'])
