import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
from .abstract import VAE
import math
import torch.optim as optim
from torchinfo import summary


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerVAE(L.LightningModule, VAE):
    def __init__(self, seq_len=800, latent_dim=128, lr=1e-3, alpha=1.0):
        super().__init__()
        self.save_hyperparameters()
        assert latent_dim % 2 == 0, "Latent dimension must be even."

        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, latent_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B, 64, 200]
        )

        self.spectrum_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, latent_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B, 64, 200]
        )

        self.pos_encoder = PositionalEncoding(latent_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=2,
                dim_feedforward=512,
                dropout=0.1,
            ),
            num_layers=2,
        )

        self.signal_decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim // 2, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2),  # [B, 1, 800]
        )

        self.spectrum_decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim // 2, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2),  # [B, 1, 800]
        )

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * (seq_len // 4), latent_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * (seq_len // 4)),
        )

    def encode(self, signal, spectrum):
        signal = self.signal_conv(signal)
        spectrum = self.spectrum_conv(spectrum)

        fused = torch.cat([signal, spectrum], dim=1)  # [B, latent_dim, 200]
        fused = fused.permute(2, 0, 1)  # [200, B, latent_dim]
        fused = self.pos_encoder(fused)
        fused = self.transformer_encoder(fused)
        fused = fused.reshape(fused.size(1), -1)  # [B, latent_dim * 200]
        latent = self.fc(fused)
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z shape: [B, latent_dim]
        decoded = self.decoder_fc(z)
        decoded = decoded.view(decoded.size(0), -1, 200)

        signal_enc, spectrum_enc = torch.chunk(decoded, 2, dim=1)
        signal_recon = self.signal_decoder(signal_enc)  # [B, 1, 800]
        spectrum_recon = self.spectrum_decoder(spectrum_enc)  # [B, 1, 800]
        return signal_recon, spectrum_recon

    def forward(self, signal, spectrum):
        mu, logvar = self.encode(signal, spectrum)
        z = self.reparameterize(mu, logvar)
        signal_recon, spectrum_recon = self.decode(z)
        return signal_recon, spectrum_recon, mu, logvar

    def train_loss(self, recon_signal, recon_spectrum, signal, spectrum, mu, logvar):
        # Reconstruction loss
        loss_signal = F.mse_loss(recon_signal, signal, reduction='mean')
        loss_spectrum = F.mse_loss(recon_spectrum, spectrum, reduction='mean')

        # KL divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + logvar -
                            mu.pow(2) - logvar.exp(), dim=1), dim=0)
        alpha = self.hparams.alpha
        return loss_signal + loss_spectrum + alpha * kl_div, loss_signal, loss_spectrum, kl_div

    def training_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        recon_signal, recon_spectrum, mu, logvar = self(signal, spectrum)
        total_loss, loss_signal, loss_spectrum, kl_div = self.train_loss(recon_signal, recon_spectrum,
                                                                         signal, spectrum, mu, logvar)
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
    model = TransformerVAE()
    summary(model, input_size=[(56, 1, 800), (56, 1, 800)])
