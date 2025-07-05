import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import torch.nn.functional as F
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
        # x shape: [batch, seq_len, d_model] when batch_first=True
        x = x + self.pe[:, :x.size(1), :]
        return x


class TrID_Signal(L.LightningModule):
    def __init__(self, seq_len=800, embedding_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Convolutional Front-end (as described in TrID paper)
        self.convolutional_frontend = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 800 -> 400
            nn.Conv1d(32, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 400 -> 200
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Transformer Encoder (2 layers as in paper)
        # Using 4 heads as mentioned in the paper (h=4)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True  # Changed to True to avoid warning
            ),
            num_layers=2,
        )

        # Convolutional Back-end (with Sigmoid as mentioned in paper)
        self.convolutional_backend = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 32,
                               kernel_size=2, stride=2),  # 200 -> 400
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),  # 400 -> 800
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),  # Final conv1D layer
            nn.Sigmoid(),  # Sigmoid activation as specified in TrID paper
        )

        # Flattening is implicit when we reshape to [B, 1, seq_len]

    def forward(self, signal):
        # Convolutional Front-end
        # Input: [B, 1, 800]
        features = self.convolutional_frontend(
            signal)  # [B, embedding_dim, 200]

        # Prepare for transformer: with batch_first=True
        # Permute to [batch, seq_len, embedding_dim]
        features = features.permute(0, 2, 1)  # [B, 200, embedding_dim]

        # Apply positional encoding (need to adjust for batch_first)
        features = self.pos_encoder(features)

        # Transformer Encoder
        encoded = self.transformer_encoder(features)  # [B, 200, embedding_dim]

        # Back to convolutional format
        encoded = encoded.permute(0, 2, 1)  # [B, embedding_dim, 200]

        # Convolutional Back-end
        reconstructed = self.convolutional_backend(encoded)  # [B, 1, 800]

        return reconstructed

    def training_step(self, batch, batch_idx):
        # For single input, batch should contain only signal
        if isinstance(batch, dict):
            signal = batch['signal']
        else:
            signal = batch

        # Forward pass
        recon_signal = self(signal)

        # MAE loss as specified in TrID paper (equation 19)
        loss = F.l1_loss(recon_signal, signal, reduction='mean')

        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # For single input, batch should contain only signal
        if isinstance(batch, dict):
            signal = batch['signal']
        else:
            signal = batch

        # Forward pass
        recon_signal = self(signal)

        # MAE loss
        loss = F.l1_loss(recon_signal, signal, reduction='mean')

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def test_step(self, batch, batch_idx):
        # For anomaly detection inference
        if isinstance(batch, dict):
            signal = batch['signal']
        else:
            signal = batch

        # Forward pass
        recon_signal = self(signal)

        # Calculate reconstruction error for each sample
        # Using MAE per sample
        recon_error = F.l1_loss(recon_signal, signal, reduction='none')
        # Average over channels and time
        recon_error = recon_error.mean(dim=[1, 2])

        return {'recon_error': recon_error}


if __name__ == '__main__':
    # Initialize model
    model = TrID_Signal(seq_len=800, embedding_dim=64)

    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Print model summary
    print("TrID_Signal Model Architecture:")
    print("="*50)
    summary(model, input_size=(8, 1, 800),
            device=device.type)  # Specify device

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 1, 800).to(
        device)  # Move input to same device
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output is in [0, 1] range due to Sigmoid
    print(
        f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Device: {device}")
