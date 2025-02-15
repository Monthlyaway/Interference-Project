import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary
from .abstract import AE

class CrossDomainAttention(nn.Module):
    """Process encoded features with spectral-temporal attention"""

    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        # Process encoded features (64 channels from encoder)
        self.time_conv = nn.Conv1d(64, embed_dim, 3, padding=1)
        self.freq_conv = nn.Conv1d(64, embed_dim, 3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, time_feat, freq_feat):
        # Input shapes: [B, 64, 200] from encoders
        B = time_feat.shape[0]

        # Project encoded features
        t_emb = self.time_conv(time_feat)  # [B, 64, 200]
        f_emb = self.freq_conv(freq_feat)

        # Reshape for attention: [200, B, 64]
        t_emb = t_emb.permute(2, 0, 1)
        f_emb = f_emb.permute(2, 0, 1)

        # Cross-domain attention
        attn_output, _ = self.attention(t_emb, f_emb, f_emb)

        # Combine features and reshape
        fused = torch.cat([t_emb, attn_output], dim=-1)  # [200, B, 128]
        return fused.permute(1, 2, 0)  # [B, 128, 200]


class CNNEncoder(nn.Module):
    """Shared encoder for time/frequency domains"""

    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.net(x)  # [B, 64, 200]


class CNNAEAttention(L.LightningModule, AE):
    def __init__(self, input_size=800, latent_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Encoders remain unchanged
        self.time_encoder = CNNEncoder()
        self.freq_encoder = CNNEncoder()

        # Attention module now processes encoded features
        self.attention = CrossDomainAttention()

        # Revised decoder with proper upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, signal, spectrum):
        # Encode to [B, 64, 200]
        time_feat = self.time_encoder(signal)
        freq_feat = self.freq_encoder(spectrum)

        # Attend to fused features [B, 128, 200]
        fused = self.attention(time_feat, freq_feat)

        # Decode back to original dimensions
        reconstructed = self.decoder(fused)  # [B, 2, 800]
        return reconstructed[:, 0, :], reconstructed[:, 1, :]

    def _shared_step(self, batch):
        signal = batch['signal']
        spectrum = batch['spectrum']

        # Reconstruction
        rec_signal, rec_spectrum = self(signal, spectrum)

        # Combined loss
        time_loss = F.mse_loss(rec_signal, signal.squeeze(1))
        freq_loss = F.mse_loss(rec_spectrum, spectrum.squeeze(1))
        total_loss = time_loss + freq_loss

        return {
            'loss': total_loss,
            'time_loss': time_loss.detach(),
            'freq_loss': freq_loss.detach()
        }

    def training_step(self, batch, batch_idx):
        losses = self._shared_step(batch)
        self.log_dict(
            {f'train/{k}': v for k, v in losses.items()}, prog_bar=True, on_step=False, on_epoch=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self._shared_step(batch)
        self.log_dict({f'val/{k}': v for k, v in losses.items()},
                      prog_bar=True, on_step=False, on_epoch=True)
        return losses['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == '__main__':
    model = CNNAEAttention()
    summary(model, input_size=[(56, 1, 800), (56, 1, 800)])
