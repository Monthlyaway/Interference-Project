import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import torch.nn.functional as F


class CNNAutoencoder(L.LightningModule):
    def __init__(self, input_length=800, latent_dim=128, learning_rate=1e-3):
        super(CNNAutoencoder, self).__init__()
        self.learning_rate = learning_rate

        # Time-Domain Encoder
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Frequency-Domain Encoder
        self.spectrum_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Fully Connected Layer for Fusion
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_length // 4), latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * (input_length // 4)),
            nn.ReLU()
        )

        # Time-Domain Decoder
        self.signal_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2)
        )

        # Frequency-Domain Decoder
        self.spectrum_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2)
        )

    def forward(self, signal, spectrum):
        # Encode Time-Domain
        signal_encoded = self.signal_encoder(signal)
        signal_encoded = signal_encoded.view(signal_encoded.size(0), -1)

        # Encode Frequency-Domain
        spectrum_encoded = self.spectrum_encoder(spectrum)
        spectrum_encoded = spectrum_encoded.view(spectrum_encoded.size(0), -1)

        # Fuse Features
        fused = torch.cat((signal_encoded, spectrum_encoded), dim=1)
        fused = self.fc(fused)

        # Decode
        decoded = self.decoder_fc(fused)
        decoded = decoded.view(signal.size(0), 32, -1)

        # Reconstruct Time-Domain
        recon_signal = self.signal_decoder(decoded)

        # Reconstruct Frequency-Domain
        recon_spectrum = self.spectrum_decoder(decoded)

        return recon_signal, recon_spectrum

    def loss_function(self, recon_signal, signal, recon_spectrum, spectrum):
        loss = F.mse_loss(recon_signal, signal, reduction="mean") + \
            F.mse_loss(recon_spectrum, spectrum, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        recon_signal, recon_spectrum = self(signal, spectrum)
        loss = self.loss_function(recon_signal, signal, recon_spectrum, spectrum)
        self.log("train/loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        signal, spectrum = batch['signal'], batch['spectrum']
        recon_signal, recon_spectrum = self(signal, spectrum)
        loss = self.loss_function(recon_signal, signal, recon_spectrum, spectrum)
        self.log("val/loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    model = CNNAutoencoder()
    summary(model, input_size=[(56, 1, 800), (56, 1, 800)])
