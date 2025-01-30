import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import torch.nn.functional as F
from .abstract import VAE

class CNNVAE(L.LightningModule, VAE):
    def __init__(self, input_length=800, latent_dim=128, lr=1e-3, alpha = 1.0):
        super().__init__()
        self.save_hyperparameters()

        self.signal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.spectrum_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * (input_length // 4), latent_dim),
            nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * (input_length // 4)),
            nn.ReLU()
        )

        self.signal_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2)
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def encode(self, signal, spectrum):
        signal_encoded = self.signal_encoder(signal)
        spectrum_encoded = self.spectrum_encoder(spectrum)

        signal_encoded = signal_encoded.reshape(signal_encoded.size(0), -1)
        spectrum_encoded = spectrum_encoded.reshape(
            spectrum_encoded.size(0), -1)

        latent = torch.cat([signal_encoded, spectrum_encoded], dim=1)
        latent = self.fc(latent)

        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.decoder_fc(z)
        decoded = decoded.view(decoded.size(0), 32, -1)

        recon_signal = self.signal_decoder(decoded)
        recon_spectrum = self.spectrum_decoder(decoded)

        return recon_signal, recon_spectrum

    def train_loss(self, recon_signal, recon_spectrum, signal, spectrum, mu, logvar):
        # Reconstruction loss
        loss_signal = F.mse_loss(recon_signal, signal, reduction='mean')
        loss_spectrum = F.mse_loss(recon_spectrum, spectrum, reduction='mean')

        # KL divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + logvar -
                            mu.pow(2) - logvar.exp(), dim=1), dim=0)
        alpha = self.hparams.alpha
        return loss_signal + loss_spectrum + alpha * kl_div, loss_signal, loss_spectrum, kl_div

    def forward(self, signal, spectrum):
        mu, logvar = self.encode(signal, spectrum)
        z = self.reparameterize(mu, logvar)
        signal, spectrum = self.decode(z)
        return signal, spectrum, mu, logvar

    def training_step(self, batch, batch_idx):
        signal = batch['signal']
        spectrum = batch['spectrum']
        recon_signal, recon_spectrum, mu, logvar = self(signal, spectrum)
        total_loss, loss_signal, loss_spectrum, kl_div = self.train_loss(
            recon_signal, recon_spectrum, signal, spectrum, mu, logvar)
        self.log_dict(
            {
                'train/total_loss': total_loss,
                'train/signal_loss': loss_signal,
                'train/spectrum_loss': loss_spectrum,
                'train/kl_div': kl_div
            }, on_step=False,  on_epoch=True
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        signal = batch['signal']
        spectrum = batch['spectrum']
        recon_signal, recon_spectrum, mu, logvar = self(signal, spectrum)
        total_loss, loss_signal, loss_spectrum, kl_div = self.train_loss(
            recon_signal, recon_spectrum, signal, spectrum, mu, logvar)
        self.log_dict(
            {
                'val/total_loss': total_loss,
                'val/signal_loss': loss_signal,
                'val/spectrum_loss': loss_spectrum,
                'val/kl_div': kl_div
            }, on_step=False,  on_epoch=True
        )
        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    model = CNNVAE()
    summary(model, input_size=[(56, 1, 800), (56, 1, 800)])
