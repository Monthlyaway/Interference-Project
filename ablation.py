import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from data_modules import BaseDataModule
from model_modules import *
import argparse
from lightning.pytorch.cli import LightningCLI
import torch

torch.set_float32_matmul_precision('medium')


class PhaseAwareConv1d(nn.Module):
    """Specialized convolution preserving phase information"""

    def __init__(self, in_channels, out_channels, kernel_size=11, padding=5):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels,
                                   kernel_size, padding=padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels,
                                   kernel_size, padding=padding)

    def forward(self, x):
        return torch.sqrt(self.conv_real(x)**2 + self.conv_imag(x)**2)


class MutualAttentionFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.query_conv = nn.Conv1d(channels, channels//8, 1)
        self.key_conv = nn.Conv1d(channels, channels//8, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # Cross-modality attention gate
        batch_size, C, L = x.size()

        Q = self.query_conv(x).view(
            batch_size, -1, L).permute(0, 2, 1)  # (B, L, C')
        K = self.key_conv(y).view(batch_size, -1, L)  # (B, C', L)

        energy = torch.bmm(Q, K)  # (B, L, L)
        attention = self.softmax(energy)

        V = self.value_conv(y).view(batch_size, -1, L)  # (B, C, L)
        out = torch.bmm(V, attention.permute(0, 2, 1))
        return x + self.gamma * out


class Conv_Att_Wave_Net(L.LightningModule):
    def __init__(self, lr=1e-3, wavelet_scales=[5, 10, 20], use_att=True):
        super().__init__()
        self.save_hyperparameters()

        # Time Encoder with phase-sensitive processing
        self.time_encoder = nn.Sequential(
            PhaseAwareConv1d(1, 32),
            nn.MaxPool1d(4),
            nn.GELU(),
            PhaseAwareConv1d(32, 64),
            nn.MaxPool1d(2),
            nn.GELU()
        )

        # Frequency Encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 11, padding=5),
            nn.MaxPool1d(4),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.MaxPool1d(2),
            nn.GELU()
        )

        if use_att:
            # Symmetric Attention Fusion
            self.temporal_attention = MutualAttentionFusion(64)
            self.spectral_attention = MutualAttentionFusion(64)

        # Multi-Path Decoder with PixelShuffle
        self.decoder = nn.Sequential(
            # Stage 1: 100 -> 200
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GELU(),

            # Stage 2: 200 -> 400
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.GELU(),

            # Stage 3: 400 -> 800
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.GELU(),

            # Final projection
            nn.Conv1d(32, 2, 3, padding=1),
            nn.Sigmoid()
        )

        if wavelet_scales != None:
            # Wavelet parameters
            self.register_buffer('wavelet_filters',
                                 self._create_wavelet_bank(wavelet_scales))

    def _create_wavelet_bank(self, scales):
        max_scale = max(scales)
        t = torch.arange(-max_scale * 2, max_scale * 2 + 1)
        filters = []
        for scale in scales:
            wavelet = torch.cos(1.75 * t / scale) * \
                torch.exp(-(t**2)/(2*(scale**2)))
            wavelet /= wavelet.norm()
            filters.append(wavelet)
        return torch.stack(filters).view(len(scales), 1, -1)

    def _wavelet_transform(self, x: torch.Tensor):
        b, c, l = x.size()
        # Convert input to (B*C, 1, L)
        x_flat = x.view(-1, 1, l)
        # Apply wavelet filters - output shape: (B*C, num_scales, L)
        x_conv = F.conv1d(x_flat, self.wavelet_filters, padding='same')
        # Make num_scales explicit
        num_scales = len(self.hparams.wavelet_scales)
        # Return as (B, C, num_scales, L)
        return x_conv.view(b, c, num_scales, l)

    def forward(self, x_time, x_freq):
        # Encoder paths
        t_feat = self.time_encoder(x_time)
        f_feat = self.freq_encoder(x_freq)

        if self.hparams.use_att:
            # Attention fusion
            t_feat = self.temporal_attention(t_feat, f_feat)
            f_feat = self.spectral_attention(f_feat, t_feat)

        # Decoder
        fused = torch.cat([t_feat, f_feat], dim=1)
        return self.decoder(fused)

    def _compute_loss(self, recon, target):
        # Time loss
        mse_loss = F.mse_loss(recon, target)

        if self.hparams.wavelet_scales is None:
            return mse_loss
        # Wavelet correlation loss
        wave_recon = self._wavelet_transform(recon)
        wave_target = self._wavelet_transform(target)
        wave_loss = 0.5 * F.mse_loss(wave_recon, wave_target)

        return mse_loss + wave_loss

    def training_step(self, batch, batch_idx):
        x_time = batch['signal']
        x_freq = batch['spectrum']
        target = torch.cat([x_time, x_freq], dim=1)

        recon = self(x_time, x_freq)
        loss = self._compute_loss(recon, target)
        self.log("train/loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_time = batch['signal']
        x_freq = batch['spectrum']
        target = torch.cat([x_time, x_freq], dim=1)

        recon = self(x_time, x_freq)
        loss = self._compute_loss(recon, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def compute_threshold(self, val_loader):
        pass

    def test_step(self, batch, batch_idx):
        x_time = batch['signal']
        x_freq = batch['spectrum']
        target = torch.cat([x_time, x_freq], dim=1)

        recon = self(x_time, x_freq)
        error = F.mse_loss(recon, target, reduction='none').mean([1, 2])
        return {'test_error': error, 'labels': batch['label']}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=1e-5)


if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project='Satelite-Interference', log_model=True)

    data_module = BaseDataModule(batch_size=128)
    data_module.prepare_data()

    # Full Net
    model_1 = Conv_Att_Wave_Net(
        lr=0.002,
        wavelet_scales=[5, 10, 20],
        use_att=True,
    )
    # No Attention
    model_2 = Conv_Att_Wave_Net(
        lr=0.002,
        wavelet_scales=[5, 10, 20],
        use_att=False,
    )
    # No Wavelet
    model_3 = Conv_Att_Wave_Net(
        lr=0.002,
        wavelet_scales=None,
        use_att=True,
    )
    # No Attention, No Wavelet
    model_4 = Conv_Att_Wave_Net(
        lr=0.002,
        wavelet_scales=None,
        use_att=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        filename='ab_1_epoch={epoch}-step={step}-val_loss={val/loss:.2f}',
        auto_insert_metric_name=False,
        save_top_k=2,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=-1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        fast_dev_run=False,
    )

    trainer.fit(model_1, data_module)
