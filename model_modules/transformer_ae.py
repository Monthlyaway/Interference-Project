import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import torch.nn.functional as F
from .abstract import AE
import math


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


class TransformerAE(L.LightningModule, AE):
    def __init__(self, seq_len=800, latent_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        assert latent_dim % 2 == 0, "Latent dimension must be even."

        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, latent_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B, 64, 200]
        )

        self.spectrum_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, latent_dim//2, kernel_size=3, padding=1),
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

    def forward(self, signal, spectrum):
        signal = self.signal_conv(signal)
        spectrum = self.spectrum_conv(spectrum)
        fused = torch.cat([signal, spectrum], dim=1)  # [B, latent_dim, 200]
        fused = fused.permute(2, 0, 1)  # [200, B, latent_dim]

        fused = self.pos_encoder(fused)
        latent = self.transformer_encoder(fused)  # [200, B, latent_dim]

        latent = latent.permute(1, 2, 0)  # [B, latent_dim, 200]
        signal_enc, spectrum_enc = torch.chunk(latent, 2, dim=1)
        signal_recon = self.signal_decoder(signal_enc)  # [B, 1, 800]
        spectrum_recon = self.spectrum_decoder(spectrum_enc)  # [B, 1, 800]
        return signal_recon, spectrum_recon

    def train_loss(self, recon_signal, signal, recon_spectrum, spectrum):
        loss_signal = F.mse_loss(recon_signal, signal, reduction='mean')
        loss_spectrum = F.mse_loss(recon_spectrum, spectrum, reduction='mean')
        return loss_signal + loss_spectrum

    def training_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        recon_signal, recon_spectrum = self(signal, spectrum)
        loss = self.train_loss(recon_signal, signal, recon_spectrum, spectrum)
        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        recon_signal, recon_spectrum = self(signal, spectrum)
        loss = self.train_loss(recon_signal, signal, recon_spectrum, spectrum)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    model = TransformerAE()
    summary(model, input_size=[(56, 1, 800), (56, 1, 800)])
