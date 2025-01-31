import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import torch.nn.functional as F
from .abstract import AE


class LinearAE(L.LightningModule, AE):
    def __init__(self, input_length=800, latent_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.signal_proj = nn.Sequential(
            nn.Linear(input_length, 512),
            nn.ReLU(),
        )

        self.spectrum_proj = nn.Sequential(
            nn.Linear(input_length, 512),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
        )

        self.signal_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_length),
        )
        self.spectrum_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_length),
        )

    def forward(self, signal, spectrum):
        signal = self.signal_proj(signal)
        spectrum = self.spectrum_proj(spectrum)
        latent = torch.cat([signal, spectrum], dim=-1)
        latent = self.encoder(latent)
        latent = self.decoder(latent)
        recon_signal = self.signal_decoder(latent)
        recon_spectrum = self.spectrum_decoder(latent)
        return recon_signal, recon_spectrum

    def train_loss(self, recon_signal, signal, recon_spectrum, spectrum):
        loss = F.mse_loss(recon_signal, signal, reduction="mean") + \
            F.mse_loss(recon_spectrum, spectrum, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        signal = signal.flatten(start_dim=1)
        spectrum = spectrum.flatten(start_dim=1)
        recon_signal, recon_spectrum = self(signal, spectrum)
        loss = self.train_loss(
            recon_signal, signal, recon_spectrum, spectrum)
        self.log("train/loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        recon_signal, recon_spectrum = self(signal, spectrum)
        loss = self.train_loss(
            recon_signal, signal, recon_spectrum, spectrum)
        self.log("val/loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    model = LinearAE()
    summary(model, input_size=[(56, 1, 800), (56, 1, 800)])
