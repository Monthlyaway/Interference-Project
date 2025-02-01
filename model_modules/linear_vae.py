import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy
from torch.optim import Adam
from .abstract import VAE


class LinearVAE(L.LightningModule, VAE):
    """Need input signal to have shape: [B, seq_len]"""

    def __init__(self, seq_len=800, latent_dim=64, lr=1e-3, alpha=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.signal_encoder = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )
        self.specturm_encoder = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )

        self.signal_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len),
        )
        self.spectrum_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len),
        )

    def encode(self, signal: torch.Tensor, spectrum: torch.Tensor):
        signal = signal.flatten(start_dim=1)
        spectrum = spectrum.flatten(start_dim=1)
        # Encode with two separate encoders
        signal = self.signal_encoder(signal)
        spectrum = self.specturm_encoder(spectrum)

        # Concatenate the two latent spaces
        latent = torch.cat([signal, spectrum], dim=-1)

        # Compute the latent space statistics
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        signal = self.signal_decoder(z)
        spectrum = self.spectrum_decoder(z)
        return signal, spectrum

    def forward(self, signal, spectrum):
        mu, logvar = self.encode(signal, spectrum)
        z = self.reparameterize(mu, logvar)
        signal, spectrum = self.decode(z)
        return signal, spectrum, mu, logvar

    def train_loss(self, recon_signal, recon_spectrum, signal, spectrum, mu, logvar):
        signal = signal.reshape(signal.shape[0], -1)
        spectrum = spectrum.reshape(signal.shape[0], -1)
        # Reconstruction loss
        loss_signal = F.mse_loss(recon_signal, signal, reduction='mean')
        loss_spectrum = F.mse_loss(recon_spectrum, spectrum, reduction='mean')

        # KL divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + logvar -
                            mu.pow(2) - logvar.exp(), dim=1), dim=0)
        alpha = self.hparams.alpha
        return loss_signal + loss_spectrum + alpha * kl_div, loss_signal, loss_spectrum, kl_div

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
            }, on_step=False,  on_epoch=True, logger=True
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
            }, on_step=False,  on_epoch=True, logger=True
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        signal = batch['signal']
        spectrum = batch['spectrum']
        recon_signal, recon_spectrum, mu, logvar = self(signal, spectrum)
        loss = self.train_loss(
            signal, spectrum, recon_signal, recon_spectrum, mu, logvar)
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    from torchviz import make_dot
    model = LinearVAE()
    graph = make_dot(model(torch.randn(64, 1, 800), torch.randn(64, 1, 800)), params=dict(model.named_parameters()))
    graph.render("linear_vae", format="png", cleanup=True)