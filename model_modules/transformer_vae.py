import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy
from torch.optim import Adam
from einops import rearrange, reduce

# Batch first is always set to True


class TransformerVAE(L.LightningModule):
    """Assume signal and spectrum have shape: [B, 1, 800].
    This is batch first notation, batch, sequence length, and number of channels."""

    def __init__(self, seq_len=800, latent_dim=64, lr=1e-3, alpha=1.0, num_layers=2, nhead=4):
        super().__init__()
        self.save_hyperparameters()

        # Project to latent space
        self.signal_in_proj = nn.Linear(1, latent_dim)
        self.spectrum_in_proj = nn.Linear(1, latent_dim)

        self.signal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=nhead),
            num_layers=num_layers
        )

        self.spectrum_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=nhead),
            num_layers=num_layers
        )

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU())
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # self.signal_decoder = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         d_model=latent_dim, num_heads=num_heads),
        #     num_layers=num_layers
        # )

        # self.spectrum_decoder = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         d_model=latent_dim, num_heads=num_heads),
        #     num_layers=num_layers
        # )

        self.signal_out_proj = nn.Sequential(
            nn.Linear(latent_dim, seq_len // 2),
            nn.GELU(),
            nn.Linear(seq_len // 2, seq_len)
        )
        self.spectrum_out_proj = nn.Sequential(
            nn.Linear(latent_dim, seq_len // 2),
            nn.GELU(),
            nn.Linear(seq_len // 2, seq_len)
        )

    def encode(self, signal, spectrum):
        signal = rearrange(signal, 'b c s -> s b c')
        spectrum = rearrange(spectrum, 'b c s -> s b c')

        # Project to latent space
        signal = self.signal_in_proj(signal)
        spectrum = self.spectrum_in_proj(spectrum)

        # Transformer encoder
        signal_encoded = self.signal_encoder(signal)
        spectrum_encoded = self.spectrum_encoder(spectrum)
        # print(f"After transformer encoder, {signal_encoded.shape=}, {
        #       spectrum_encoded.shape=}")  # [800, B, E]

        # Reduce the dimension of Length(800) to 1
        signal_encoded = reduce(signal_encoded, 's b e -> b e', 'mean')
        spectrum_encoded = reduce(spectrum_encoded, 's b e -> b e', 'mean')

        # Concatenate the two latent spaces
        latent = torch.cat([signal_encoded, spectrum_encoded], dim=-1)
        latent = self.latent_proj(latent)

        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)

        # latent is used in decoder transformer to recontruct
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        signal = self.signal_out_proj(z)
        spectrum = self.spectrum_out_proj(z)
        return signal, spectrum

    def train_loss(self, recon_signal, recon_spectrum, signal, spectrum, mu, logvar):
        # Reconstruction loss
        recon_signal = recon_signal.unsqueeze(1)
        recon_spectrum = recon_spectrum.unsqueeze(1)
        loss_signal = F.mse_loss(recon_signal, signal, reduction='mean')
        loss_spectrum = F.mse_loss(recon_spectrum, spectrum, reduction='mean')

        # KL divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + logvar -
                            mu.pow(2) - logvar.exp(), dim=1), dim=0)
        alpha = self.hparams.alpha
        return loss_signal + loss_spectrum + alpha * kl_div, loss_signal, loss_spectrum, kl_div

    def forward(self, signal, spectrum):
        mu, logvar = self.encode(
            signal, spectrum)
        z = self.reparameterize(mu, logvar)
        recon_signal, recon_spectrum = self.decode(z)
        return recon_signal, recon_spectrum, mu, logvar

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
        return Adam(self.parameters(), lr=self.hparams.lr)
