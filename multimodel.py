import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import time
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from ptflops import get_model_complexity_info

# ----------------------------
# Configurations
# ----------------------------

# Configure Matplotlib to support English titles and legends
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.unicode_minus': False  # To display negative signs correctly
})

# ----------------------------
# Data Loading and Preprocessing
# ----------------------------

def load_mat_data(file_path, data_key_contains='Data'):
    """Load data from a .mat file."""
    data = scipy.io.loadmat(file_path)
    for key in data:
        if data_key_contains in key:
            return data[key]
    raise ValueError(f"Data variable containing '{data_key_contains}' not found in {file_path}.")

def load_mat_labels(file_path, label_key_contains='Labels'):
    """Load labels from a .mat file."""
    data = scipy.io.loadmat(file_path)
    for key in data:
        if label_key_contains in key:
            return data[key].flatten()
    raise ValueError(f"Label variable containing '{label_key_contains}' not found in {file_path}.")

class TimeFrequencyDataset(Dataset):
    """Custom Dataset for time and frequency domain data."""
    def __init__(self, signal, spectrum, labels=None):
        self.signal = signal
        self.spectrum = spectrum
        self.labels = labels

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx):
        data = {
            'signal': self.signal[idx],
            'spectrum': self.spectrum[idx]
        }
        if self.labels is not None:
            data['label'] = self.labels[idx]
        return data

# ----------------------------
# Model Definitions
# ----------------------------

# Transformer-Based Multi-Modal Autoencoder
class MultiModalTransformerAutoencoder(nn.Module):
    def __init__(self, input_length=800, latent_dim=64, nhead=2, num_encoder_layers=1, 
                 num_decoder_layers=1, dim_feedforward=128, dropout=0.1):
        super(MultiModalTransformerAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Input Projections for Signal and Spectrum
        self.signal_input_proj = nn.Linear(1, latent_dim)
        self.spectrum_input_proj = nn.Linear(1, latent_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Fusion Layer
        self.latent_proj = nn.Linear(latent_dim * 2, latent_dim)

        # Output Projections
        self.signal_output_proj = nn.Linear(latent_dim, 1)
        self.spectrum_output_proj = nn.Linear(latent_dim, 1)

    def forward(self, signal, spectrum):
        # Transpose for Transformer: (Seq_Length, Batch_Size, Features)
        signal = signal.permute(2, 0, 1)  # (800, Batch, 1)
        spectrum = spectrum.permute(2, 0, 1)

        # Input Projection
        signal = self.signal_input_proj(signal)  # (800, Batch, Latent)
        spectrum = self.spectrum_input_proj(spectrum)

        # Transformer Encoding
        encoded_signal = self.transformer_encoder(signal)  # (800, Batch, Latent)
        encoded_spectrum = self.transformer_encoder(spectrum)

        # Pooling (Mean over sequence length)
        encoded_signal = torch.mean(encoded_signal, dim=0)  # (Batch, Latent)
        encoded_spectrum = torch.mean(encoded_spectrum, dim=0)

        # Fusion
        fused = torch.cat((encoded_signal, encoded_spectrum), dim=1)  # (Batch, 2*Latent)
        fused = self.latent_proj(fused)  # (Batch, Latent)

        # Decoder Input (Zero sequence)
        tgt = torch.zeros(800, signal.size(1), self.latent_dim).to(signal.device)  # (800, Batch, Latent)

        # Transformer Decoding
        decoded = self.transformer_decoder(tgt, fused.unsqueeze(0))  # (800, Batch, Latent)

        # Output Projection
        recon_signal = self.signal_output_proj(decoded)  # (800, Batch, 1)
        recon_spectrum = self.spectrum_output_proj(decoded)

        # Transpose back to (Batch, Features, Seq_Length)
        recon_signal = recon_signal.permute(1, 2, 0)
        recon_spectrum = recon_spectrum.permute(1, 2, 0)

        return recon_signal, recon_spectrum

# CNN-Based Multi-Modal Autoencoder
class CNNMultiModalAutoencoder(nn.Module):
    def __init__(self, input_length=800, latent_dim=128):
        super(CNNMultiModalAutoencoder, self).__init__()
        
        # Time-Domain Encoder
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2)
        )
        
        # Frequency-Domain Encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2)
        )
        
        # Fully Connected Layer for Fusion
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_length // 4), latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * (input_length // 4)),
            nn.ReLU(True)
        )
        
        # Time-Domain Decoder
        self.time_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
        # Frequency-Domain Decoder
        self.freq_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, time, freq):
        # Encode Time-Domain
        time_encoded = self.time_encoder(time)  # (Batch, 32, 200)
        time_encoded = time_encoded.view(time_encoded.size(0), -1)  # (Batch, 32*200)
        
        # Encode Frequency-Domain
        freq_encoded = self.freq_encoder(freq)  # (Batch, 32, 200)
        freq_encoded = freq_encoded.view(freq_encoded.size(0), -1)  # (Batch, 32*200)
        
        # Fuse Features
        fused = torch.cat((time_encoded, freq_encoded), dim=1)  # (Batch, 64*200)
        fused = self.fc(fused)  # (Batch, Latent)
        
        # Decode
        decoded = self.decoder_fc(fused)  # (Batch, 32*200)
        decoded = decoded.view(time.size(0), 32, -1)  # (Batch, 32, 200)
        
        # Reconstruct Time-Domain
        recon_time = self.time_decoder(decoded)  # (Batch, 1, 800)
        
        # Reconstruct Frequency-Domain
        recon_freq = self.freq_decoder(decoded)  # (Batch, 1, 800)
        
        return recon_time, recon_freq

# Single-Modal Time-Domain Autoencoder
class TimeAutoencoder(nn.Module):
    def __init__(self, input_dim=800, latent_dim=64):
        super(TimeAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Assuming normalized input
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Single-Modal Frequency-Domain Autoencoder
class FrequencyAutoencoder(nn.Module):
    def __init__(self, input_dim=800, latent_dim=64):
        super(FrequencyAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Assuming normalized input
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ----------------------------
# Baseline Methods
# ----------------------------

def pca_anomaly_detection(train_data, test_data, threshold_percentile=95):
    """
    Perform PCA-based anomaly detection.

    Parameters:
        train_data (numpy.ndarray): Training data (n_samples, n_features).
        test_data (numpy.ndarray): Testing data (n_samples, n_features).
        threshold_percentile (float): Percentile to determine anomaly threshold.

    Returns:
        reconstruction_errors (numpy.ndarray): Reconstruction errors for test data.
        predicted_labels (numpy.ndarray): Predicted anomaly labels (0: normal, 1: anomaly).
        threshold (float): Anomaly detection threshold.
    """
    # Standardize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% variance
    pca.fit(train_scaled)
    
    # Reconstruct test data
    test_reduced = pca.transform(test_scaled)
    test_reconstructed = pca.inverse_transform(test_reduced)
    
    # Calculate reconstruction error
    reconstruction_errors = np.mean((test_scaled - test_reconstructed) ** 2, axis=1)
    
    # Determine threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Predict labels
    predicted_labels = (reconstruction_errors > threshold).astype(int)
    
    return reconstruction_errors, predicted_labels, threshold

def isolation_forest_anomaly_detection(train_data, test_data, contamination=0.05):
    """
    Perform Isolation Forest-based anomaly detection.

    Parameters:
        train_data (numpy.ndarray): Training data (n_samples, n_features).
        test_data (numpy.ndarray): Testing data (n_samples, n_features).
        contamination (float): Expected proportion of anomalies.

    Returns:
        scores (numpy.ndarray): Anomaly scores for test data.
        predicted_labels (numpy.ndarray): Predicted anomaly labels (0: normal, 1: anomaly).
    """
    # Standardize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(train_scaled)
    
    # Predict anomalies
    predicted_labels = iso_forest.predict(test_scaled)
    predicted_labels = (predicted_labels == -1).astype(int)  # Convert to 0 and 1
    
    # Get anomaly scores
    scores = -iso_forest.decision_function(test_scaled)
    
    return scores, predicted_labels

# ----------------------------
# Training and Evaluation Functions
# ----------------------------

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(model, input_signal, input_spectrum, device):
    """
    Compute the number of FLOPs for a given model.

    Parameters:
        model (nn.Module): The PyTorch model.
        input_signal (torch.Tensor): Sample input for signal (batch_size, channels, length).
        input_spectrum (torch.Tensor or None): Sample input for spectrum (batch_size, channels, length).
        device (torch.device): Device to perform computation on.

    Returns:
        float: Number of FLOPs.
    """
    try:
        if isinstance(model, MultiModalTransformerAutoencoder) or isinstance(model, CNNMultiModalAutoencoder):
            # For multi-modal models, ptflops may not support multiple inputs directly
            # Hence, compute FLOPs separately for each input and sum them
            signal_flops, _ = get_model_complexity_info(
                model.signal_input_proj, 
                (1, input_signal.shape[2]),
                as_strings=False,
                print_per_layer_stat=False
            )
            spectrum_flops, _ = get_model_complexity_info(
                model.spectrum_input_proj, 
                (1, input_spectrum.shape[2]),
                as_strings=False,
                print_per_layer_stat=False
            )
            total_flops = 2 * (signal_flops + spectrum_flops)
        else:
            # For single-modal models
            if isinstance(model, (TimeAutoencoder, FrequencyAutoencoder)):
                flops, _ = get_model_complexity_info(
                    model, 
                    (input_signal.shape[1], input_signal.shape[2]),
                    as_strings=False,
                    print_per_layer_stat=False
                )
                total_flops = 2 * flops  # MACs to FLOPs
            else:
                total_flops = 'Unsupported Model'
        return total_flops
    except Exception as e:
        print(f"Failed to compute FLOPs for model {model.__class__.__name__}: {e}")
        return 'Unable to compute'

def train_autoencoder(model, train_loader, val_loader, device, num_epochs=10, patience=3, learning_rate=0.001):
    """
    Train an autoencoder model with early stopping.

    Parameters:
        model (nn.Module): The autoencoder model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train on.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        list: Training MSE losses per epoch.
        list: Validation MSE losses per epoch.
        list: Training MAE losses per epoch.
        list: Validation MAE losses per epoch.
        float: Total training time in seconds.
    """
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_loss = np.inf
    patience_counter = 0

    train_losses_mse = []
    val_losses_mse = []
    train_losses_mae = []
    val_losses_mae = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss_mse = 0.0
        running_loss_mae = 0.0

        for batch in train_loader:
            if isinstance(model, MultiModalTransformerAutoencoder) or isinstance(model, CNNMultiModalAutoencoder):
                # Multi-modal models
                signal = batch['signal'].to(device)
                spectrum = batch['spectrum'].to(device)
                optimizer.zero_grad()

                with autocast():
                    recon_signal, recon_spectrum = model(signal, spectrum)
                    loss_signal_mse = criterion_mse(recon_signal, signal)
                    loss_spectrum_mse = criterion_mse(recon_spectrum, spectrum)
                    loss_signal_mae = criterion_mae(recon_signal, signal)
                    loss_spectrum_mae = criterion_mae(recon_spectrum, spectrum)
                    loss_mse = loss_signal_mse + loss_spectrum_mse
                    loss_mae = loss_signal_mae + loss_spectrum_mae

                scaler.scale(loss_mse).backward()
                scaler.scale(loss_mae).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss_mse += loss_mse.item() * signal.size(0)
                running_loss_mae += loss_mae.item() * signal.size(0)
            else:
                # Single-modal models
                if 'signal' in batch and isinstance(model, TimeAutoencoder):
                    data = batch['signal'].squeeze(1).to(device)
                elif 'spectrum' in batch and isinstance(model, FrequencyAutoencoder):
                    data = batch['spectrum'].squeeze(1).to(device)
                else:
                    raise ValueError("Unsupported model or data format.")

                optimizer.zero_grad()

                with autocast():
                    recon = model(data)
                    loss_mse = criterion_mse(recon, data)
                    loss_mae = criterion_mae(recon, data)

                scaler.scale(loss_mse).backward()
                scaler.scale(loss_mae).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss_mse += loss_mse.item() * data.size(0)
                running_loss_mae += loss_mae.item() * data.size(0)

        epoch_loss_mse = running_loss_mse / len(train_loader.dataset)
        epoch_loss_mae = running_loss_mae / len(train_loader.dataset)
        train_losses_mse.append(epoch_loss_mse)
        train_losses_mae.append(epoch_loss_mae)

        # Validation
        model.eval()
        val_running_loss_mse = 0.0
        val_running_loss_mae = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(model, MultiModalTransformerAutoencoder) or isinstance(model, CNNMultiModalAutoencoder):
                    # Multi-modal models
                    signal = batch['signal'].to(device)
                    spectrum = batch['spectrum'].to(device)

                    with autocast():
                        recon_signal, recon_spectrum = model(signal, spectrum)
                        loss_signal_mse = criterion_mse(recon_signal, signal)
                        loss_spectrum_mse = criterion_mse(recon_spectrum, spectrum)
                        loss_signal_mae = criterion_mae(recon_signal, signal)
                        loss_spectrum_mae = criterion_mae(recon_spectrum, spectrum)
                        loss_mse = loss_signal_mse + loss_spectrum_mse
                        loss_mae = loss_signal_mae + loss_spectrum_mae
                else:
                    # Single-modal models
                    if 'signal' in batch and isinstance(model, TimeAutoencoder):
                        data = batch['signal'].squeeze(1).to(device)
                    elif 'spectrum' in batch and isinstance(model, FrequencyAutoencoder):
                        data = batch['spectrum'].squeeze(1).to(device)
                    else:
                        raise ValueError("Unsupported model or data format.")

                    with autocast():
                        recon = model(data)
                        loss_mse = criterion_mse(recon, data)
                        loss_mae = criterion_mae(recon, data)

                val_running_loss_mse += loss_mse.item() * (signal.size(0) if isinstance(model, MultiModalTransformerAutoencoder) or isinstance(model, CNNMultiModalAutoencoder) else data.size(0))
                val_running_loss_mae += loss_mae.item() * (signal.size(0) if isinstance(model, MultiModalTransformerAutoencoder) or isinstance(model, CNNMultiModalAutoencoder) else data.size(0))

        val_epoch_loss_mse = val_running_loss_mse / len(val_loader.dataset)
        val_epoch_loss_mae = val_running_loss_mae / len(val_loader.dataset)
        val_losses_mse.append(val_epoch_loss_mse)
        val_losses_mae.append(val_epoch_loss_mae)

        print(f'Epoch {epoch+1}/{num_epochs} | Train MSE: {epoch_loss_mse:.6f} | Val MSE: {val_epoch_loss_mse:.6f} | Train MAE: {epoch_loss_mae:.6f} | Val MAE: {val_epoch_loss_mae:.6f}')

        # Early Stopping based on MSE
        if val_epoch_loss_mse < best_val_loss:
            best_val_loss = val_epoch_loss_mse
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Clear cache
        torch.cuda.empty_cache()

    training_time = time.time() - start_time

    # Load best model
    model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
    return train_losses_mse, val_losses_mse, train_losses_mae, val_losses_mae, training_time

def plot_training_mae(loss_history):
    """
    Plot training and validation MAE loss curves for all models.

    Parameters:
        loss_history (dict): Dictionary containing MAE loss histories for each model.
    """
    plt.figure(figsize=(12, 8))
    for model_name, losses in loss_history.items():
        epochs = range(1, len(losses['train_mae']) + 1)
        plt.plot(epochs, losses['train_mae'], label=f'{model_name} Train MAE')
        plt.plot(epochs, losses['val_mae'], linestyle='--', label=f'{model_name} Val MAE')

    plt.xlabel('Epochs')
    plt.ylabel('MAE Loss')
    plt.title('Training and Validation MAE Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on test data.

    Parameters:
        model (nn.Module): The trained autoencoder model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to perform evaluation on.

    Returns:
        dict: Evaluation metrics including reconstruction errors, predicted labels, threshold, ROC AUC, confusion matrix, and classification report.
    """
    model.eval()
    reconstruction_errors = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(model, MultiModalTransformerAutoencoder) or isinstance(model, CNNMultiModalAutoencoder):
                # Multi-modal models
                signal = batch['signal'].to(device)
                spectrum = batch['spectrum'].to(device)
                labels = batch['label'].cpu().numpy()

                recon_signal, recon_spectrum = model(signal, spectrum)
                # Calculate reconstruction error per sample
                error_signal = torch.mean((recon_signal - signal) ** 2, dim=[1,2]).cpu().numpy()
                error_spectrum = torch.mean((recon_spectrum - spectrum) ** 2, dim=[1,2]).cpu().numpy()
                total_error = error_signal + error_spectrum
                reconstruction_errors.extend(total_error)
                all_true_labels.extend(labels)
            else:
                # Single-modal models
                if 'signal' in batch and isinstance(model, TimeAutoencoder):
                    data = batch['signal'].squeeze(1).to(device)
                elif 'spectrum' in batch and isinstance(model, FrequencyAutoencoder):
                    data = batch['spectrum'].squeeze(1).to(device)
                else:
                    raise ValueError("Unsupported model or data format.")

                labels = batch['label'].cpu().numpy()

                recon = model(data)
                loss = torch.mean((recon - data) ** 2, dim=1).cpu().numpy()
                reconstruction_errors.extend(loss)
                all_true_labels.extend(labels)
    
    reconstruction_errors = np.array(reconstruction_errors)
    all_true_labels = np.array(all_true_labels)

    # Determine threshold (e.g., 95th percentile)
    threshold = np.percentile(reconstruction_errors, 95)
    print(f'Anomaly detection threshold set at: {threshold:.6f}')

    # Predict labels
    predicted_labels = (reconstruction_errors > threshold).astype(int)

    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(all_true_labels, reconstruction_errors)
    roc_auc_val = auc(fpr, tpr)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_true_labels, predicted_labels)
    cr = classification_report(all_true_labels, predicted_labels, target_names=['Interference-free', 'Interference'])

    return {
        'reconstruction_errors': reconstruction_errors,
        'predicted_labels': predicted_labels,
        'threshold': threshold,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc_val,
        'confusion_matrix': cm,
        'classification_report': cr
    }

# ----------------------------
# Efficiency Metrics Functions
# ----------------------------

def plot_efficiency_comparison(efficiency):
    """Plot comparison of models based on efficiency metrics."""
    # Convert efficiency data to DataFrame
    data = []
    for model, metrics in efficiency.items():
        data.append({
            'Model': model,
            'Parameter Count': metrics['Parameters'],
            'Training Time (s)': metrics['Training Time (s)'],
            'FLOPs': metrics['FLOPs']
        })
    df = pd.DataFrame(data)

    # 处理 'Parameter Count' 列
    df['Parameter Count'] = pd.to_numeric(df['Parameter Count'], errors='coerce')
    df_params = df.dropna(subset=['Parameter Count']).copy()
    df_params['Parameter Count'] = df_params['Parameter Count'].astype(int)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Parameter Count', data=df_params, palette='viridis')
    plt.xlabel('Model')
    plt.ylabel('Number of Parameters')
    plt.title('Comparison of Model Parameter Counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 处理 'Training Time (s)' 列
    df['Training Time (s)'] = pd.to_numeric(df['Training Time (s)'], errors='coerce')
    df_time = df.dropna(subset=['Training Time (s)']).copy()
    df_time['Training Time (s)'] = df_time['Training Time (s)'].astype(float)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Training Time (s)', data=df_time, palette='magma')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Comparison of Model Training Times')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 处理 'FLOPs' 列
    df['FLOPs'] = pd.to_numeric(df['FLOPs'], errors='coerce')
    df_flops = df.dropna(subset=['FLOPs']).copy()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='FLOPs', data=df_flops, palette='coolwarm')
    plt.xlabel('Model')
    plt.ylabel('FLOPs')
    plt.title('Comparison of Model Computational Complexity (FLOPs)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# 确保只有一个 plot_comparative_roc 函数定义
def plot_comparative_roc(results):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10,8))
    for model_name, metrics in results.items():
        plt.plot(metrics['fpr'], metrics['tpr'], lw=2, label=f'{model_name} (AUC = {metrics["roc_auc"]:.2f})')
    
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([-0.01,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_comparative_confusion_matrices(results):
    """
    Plot confusion matrices for all models side by side for comparison.

    Parameters:
        results (dict): Dictionary containing confusion matrices for each model.
    """
    num_models = len(results)
    plt.figure(figsize=(5 * num_models, 6))
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = metrics['confusion_matrix']
        plt.subplot(1, num_models, idx+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Interference-free', 'Interference'], 
                    yticklabels=['Interference-free', 'Interference'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.show()

def plot_reconstruction_errors_distribution(results, test_labels):
    """Plot distribution of reconstruction errors or anomaly scores."""
    plt.figure(figsize=(12,8))
    for model_name, metrics in results.items():
        if model_name in ['PCA', 'Isolation_Forest']:
            scores = metrics['reconstruction_errors'] if 'reconstruction_errors' in metrics else metrics['scores']
            sns.histplot(scores[test_labels == 0], color='blue', label=f'{model_name} Interference-free', kde=True, stat="density", bins=50, alpha=0.5)
            sns.histplot(scores[test_labels == 1], color='red', label=f'{model_name} Interference', kde=True, stat="density", bins=50, alpha=0.5)
        else:
            errors = metrics['reconstruction_errors']
            sns.histplot(errors[test_labels == 0], color='green', label=f'{model_name} Interference-free', kde=True, stat="density", bins=50, alpha=0.5)
            sns.histplot(errors[test_labels == 1], color='orange', label=f'{model_name} Interference', kde=True, stat="density", bins=50, alpha=0.5)
    
    # Example: Plot threshold for Transformer
    if 'Transformer' in results:
        plt.axvline(results['Transformer']['threshold'], color='black', linestyle='--', label='Transformer Threshold')
    
    plt.xlabel('Reconstruction Error / Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors / Anomaly Scores')
    plt.legend()
    plt.show()

# ----------------------------
# Main Training and Evaluation Function
# ----------------------------

def run_all_models_and_evaluate(train_loader, val_loader, test_loader, device, test_labels):
    """
    Train and evaluate all models, collect performance and efficiency metrics.

    Parameters:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to train/evaluate on.
        test_labels (numpy.ndarray): True labels for the test data.

    Returns:
        dict: Evaluation results for each model.
        dict: Efficiency metrics for each model.
        dict: Training and validation MAE losses for each model.
    """
    results = {}
    efficiency = {}
    loss_history = {}

    # 1. Transformer-Based Multi-Modal Autoencoder
    print("Training Transformer-Based Multi-Modal Autoencoder...")
    transformer_model = MultiModalTransformerAutoencoder(
        input_length=800,
        latent_dim=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)
    train_losses_mse, val_losses_mse, train_losses_mae, val_losses_mae, training_time = train_autoencoder(transformer_model, train_loader, val_loader, device, num_epochs=10, patience=3)
    results['Transformer'] = evaluate_model(transformer_model, test_loader, device)
    efficiency['Transformer'] = {
        'Parameters': count_parameters(transformer_model),
        'Training Time (s)': training_time,
        'FLOPs': compute_flops(transformer_model, torch.randn(1, 1, 800).to(device), 
                               torch.randn(1, 1, 800).to(device), device)
    }
    loss_history['Transformer'] = {
        'train_mae': train_losses_mae,
        'val_mae': val_losses_mae
    }

    # 2. CNN-Based Multi-Modal Autoencoder
    print("\nTraining CNN-Based Multi-Modal Autoencoder...")
    cnn_model = CNNMultiModalAutoencoder(input_length=800, latent_dim=128).to(device)
    train_losses_cnn_mse, val_losses_cnn_mse, train_losses_cnn_mae, val_losses_cnn_mae, training_time_cnn = train_autoencoder(cnn_model, train_loader, val_loader, device, num_epochs=10, patience=3)
    results['CNN_MultiModal'] = evaluate_model(cnn_model, test_loader, device)
    efficiency['CNN_MultiModal'] = {
        'Parameters': count_parameters(cnn_model),
        'Training Time (s)': training_time_cnn,
        'FLOPs': compute_flops(cnn_model, torch.randn(1, 1, 800).to(device), 
                               torch.randn(1, 1, 800).to(device), device)
    }
    loss_history['CNN_MultiModal'] = {
        'train_mae': train_losses_cnn_mae,
        'val_mae': val_losses_cnn_mae
    }

    # 3. Single-Modal Time-Domain Autoencoder
    print("\nTraining Single-Modal Time-Domain Autoencoder...")
    time_autoencoder = TimeAutoencoder(input_dim=800, latent_dim=64).to(device)
    train_losses_time_mse, val_losses_time_mse, train_losses_time_mae, val_losses_time_mae, training_time_time = train_autoencoder(time_autoencoder, train_loader, val_loader, device, num_epochs=10, patience=3)
    results['Time_Autoencoder'] = evaluate_model(time_autoencoder, test_loader, device)
    efficiency['Time_Autoencoder'] = {
        'Parameters': count_parameters(time_autoencoder),
        'Training Time (s)': training_time_time,
        'FLOPs': compute_flops(time_autoencoder, torch.randn(1, 800).to(device), 
                               None, device)
    }
    loss_history['Time_Autoencoder'] = {
        'train_mae': train_losses_time_mae,
        'val_mae': val_losses_time_mae
    }

    # 4. Single-Modal Frequency-Domain Autoencoder
    print("\nTraining Single-Modal Frequency-Domain Autoencoder...")
    freq_autoencoder = FrequencyAutoencoder(input_dim=800, latent_dim=64).to(device)
    train_losses_freq_mse, val_losses_freq_mse, train_losses_freq_mae, val_losses_freq_mae, training_time_freq = train_autoencoder(freq_autoencoder, train_loader, val_loader, device, num_epochs=10, patience=3)
    results['Frequency_Autoencoder'] = evaluate_model(freq_autoencoder, test_loader, device)
    efficiency['Frequency_Autoencoder'] = {
        'Parameters': count_parameters(freq_autoencoder),
        'Training Time (s)': training_time_freq,
        'FLOPs': compute_flops(freq_autoencoder, torch.randn(1, 800).to(device), 
                               None, device)
    }
    loss_history['Frequency_Autoencoder'] = {
        'train_mae': train_losses_freq_mae,
        'val_mae': val_losses_freq_mae
    }

    # 5. PCA-Based Anomaly Detection
    print("\nRunning PCA-Based Anomaly Detection...")
    # Extract training and testing data
    train_signals = torch.cat([batch['signal'] for batch in train_loader], dim=0).cpu().numpy()
    train_spectrums = torch.cat([batch['spectrum'] for batch in train_loader], dim=0).cpu().numpy()
    test_signals = torch.cat([batch['signal'] for batch in test_loader], dim=0).cpu().numpy()
    test_spectrums = torch.cat([batch['spectrum'] for batch in test_loader], dim=0).cpu().numpy()

    # Concatenate multi-modal data
    train_flat_pca = np.hstack((train_signals.reshape(train_signals.shape[0], -1),
                                train_spectrums.reshape(train_spectrums.shape[0], -1)))
    test_flat_pca = np.hstack((test_signals.reshape(test_signals.shape[0], -1),
                               test_spectrums.reshape(test_spectrums.shape[0], -1)))

    pca_reconstruction_errors, pca_predicted_labels, pca_threshold = pca_anomaly_detection(train_flat_pca, test_flat_pca)

    # ROC AUC
    fpr_pca, tpr_pca, _ = roc_curve(test_labels, pca_reconstruction_errors)
    roc_auc_pca = auc(fpr_pca, tpr_pca)

    # Confusion Matrix and Report
    cm_pca = confusion_matrix(test_labels, pca_predicted_labels)
    cr_pca = classification_report(test_labels, pca_predicted_labels, target_names=['Interference-free', 'Interference'])

    results['PCA'] = {
        'reconstruction_errors': pca_reconstruction_errors,
        'predicted_labels': pca_predicted_labels,
        'threshold': pca_threshold,
        'fpr': fpr_pca,
        'tpr': tpr_pca,
        'roc_auc': roc_auc_pca,
        'confusion_matrix': cm_pca,
        'classification_report': cr_pca
    }
    efficiency['PCA'] = {
        'Parameters': 'N/A (Baseline)',
        'Training Time (s)': 'N/A (Baseline)',
        'FLOPs': 'N/A (Baseline)'
    }

    # 6. Isolation Forest
    print("\nRunning Isolation Forest Anomaly Detection...")
    iso_scores, iso_predicted_labels = isolation_forest_anomaly_detection(train_flat_pca, test_flat_pca)

    # ROC AUC
    fpr_iso, tpr_iso, _ = roc_curve(test_labels, iso_scores)
    roc_auc_iso = auc(fpr_iso, tpr_iso)

    # Confusion Matrix and Report
    cm_iso = confusion_matrix(test_labels, iso_predicted_labels)
    cr_iso = classification_report(test_labels, iso_predicted_labels, target_names=['Interference-free', 'Interference'])

    results['Isolation_Forest'] = {
        'scores': iso_scores,
        'predicted_labels': iso_predicted_labels,
        'fpr': fpr_iso,
        'tpr': tpr_iso,
        'roc_auc': roc_auc_iso,
        'confusion_matrix': cm_iso,
        'classification_report': cr_iso
    }
    efficiency['Isolation_Forest'] = {
        'Parameters': 'N/A (Baseline)',
        'Training Time (s)': 'N/A (Baseline)',
        'FLOPs': 'N/A (Baseline)'
    }

    return results, efficiency, loss_history

# ----------------------------
# Plotting Functions
# ----------------------------

def plot_comparative_confusion_matrices(results):
    """Plot confusion matrices for all models."""
    num_models = len(results)
    plt.figure(figsize=(5 * num_models, 6))
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = metrics['confusion_matrix']
        plt.subplot(1, num_models, idx+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Interference-free', 'Interference'], 
                    yticklabels=['Interference-free', 'Interference'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.show()



# ----------------------------
# Main Function
# ----------------------------

def main():
    # Specify file paths
    signal_train_path = '1_Signal_Datasets/Signal_TrainingData.mat'
    signal_val_path = '1_Signal_Datasets/Signal_ValidationData.mat'
    signal_test_path = '1_Signal_Datasets/Signal_TestData.mat'

    spectrum_train_path = '2_Spectrm_Datasets/Spectrum_TrainingData.mat'
    spectrum_val_path = '2_Spectrm_Datasets/Spectrum_ValidationData.mat'
    spectrum_test_path = '2_Spectrm_Datasets/Spectrum_TestData.mat'

    signal_test_labels_path = '1_Signal_Datasets/Signal_TestLabels.mat'
    spectrum_test_labels_path = '2_Spectrm_Datasets/Spectrum_TestLabels.mat'

    # Verify all files exist
    file_paths = [
        signal_train_path, signal_val_path, signal_test_path,
        spectrum_train_path, spectrum_val_path, spectrum_test_path,
        signal_test_labels_path, spectrum_test_labels_path
    ]

    for path in file_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"File not found: {path}")

    # Load data
    signal_train = load_mat_data(signal_train_path)       # Shape: (11500, 800)
    signal_val = load_mat_data(signal_val_path)           # Shape: (1302, 800)
    signal_test = load_mat_data(signal_test_path)         # Shape: (4470, 800)

    spectrum_train = load_mat_data(spectrum_train_path)   # Shape: (11500, 800)
    spectrum_val = load_mat_data(spectrum_val_path)       # Shape: (1302, 800)
    spectrum_test = load_mat_data(spectrum_test_path)     # Shape: (4470, 800)

    # Load test labels
    signal_test_labels = load_mat_labels(signal_test_labels_path)      # Shape: (4470,)
    spectrum_test_labels = load_mat_labels(spectrum_test_labels_path)  # Shape: (4470,)

    # Ensure test labels are consistent
    assert np.array_equal(signal_test_labels, spectrum_test_labels), "Test labels for signal and spectrum data do not match."
    test_labels = signal_test_labels

    # Data Preprocessing
    # Standardize data
    scaler_signal = StandardScaler()
    scaler_spectrum = StandardScaler()

    # Fit on training data and transform
    signal_train = scaler_signal.fit_transform(signal_train)
    signal_val = scaler_signal.transform(signal_val)
    signal_test = scaler_signal.transform(signal_test)

    spectrum_train = scaler_spectrum.fit_transform(spectrum_train)
    spectrum_val = scaler_spectrum.transform(spectrum_val)
    spectrum_test = scaler_spectrum.transform(spectrum_test)

    # Reshape for PyTorch (Batch_Size, Channels, Length)
    signal_train = signal_train[:, np.newaxis, :]    # Shape: (11500, 1, 800)
    signal_val = signal_val[:, np.newaxis, :]        # Shape: (1302, 1, 800)
    signal_test = signal_test[:, np.newaxis, :]      # Shape: (4470, 1, 800)

    spectrum_train = spectrum_train[:, np.newaxis, :]  # Shape: (11500, 1, 800)
    spectrum_val = spectrum_val[:, np.newaxis, :]      # Shape: (1302, 1, 800)
    spectrum_test = spectrum_test[:, np.newaxis, :]    # Shape: (4470, 1, 800)

    # Convert to PyTorch tensors
    signal_train = torch.tensor(signal_train, dtype=torch.float32)
    signal_val = torch.tensor(signal_val, dtype=torch.float32)
    signal_test = torch.tensor(signal_test, dtype=torch.float32)

    spectrum_train = torch.tensor(spectrum_train, dtype=torch.float32)
    spectrum_val = torch.tensor(spectrum_val, dtype=torch.float32)
    spectrum_test = torch.tensor(spectrum_test, dtype=torch.float32)

    # Create Datasets
    train_dataset = TimeFrequencyDataset(signal_train, spectrum_train, labels=None)  # No labels
    val_dataset = TimeFrequencyDataset(signal_val, spectrum_val, labels=None)        # No labels
    test_dataset = TimeFrequencyDataset(signal_test, spectrum_test, labels=test_labels)  # With labels

    # Create DataLoaders
    batch_size = 8  # Adjust based on GPU capacity
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Run all models and evaluate
    results, efficiency, loss_history = run_all_models_and_evaluate(train_loader, val_loader, test_loader, device, test_labels)

    # Generate Plots
    plot_comparative_roc(results) 
    plot_comparative_confusion_matrices(results)
    plot_reconstruction_errors_distribution(results, test_labels)
    plot_efficiency_comparison(efficiency)
    plot_training_mae(loss_history)  # 绘制 MAE 训练损失图

    # Save efficiency metrics to CSV
    efficiency_df = pd.DataFrame([
        {'Model': model, 'Parameter Count': metrics['Parameters'], 
         'Training Time (s)': metrics['Training Time (s)'], 'FLOPs': metrics['FLOPs']}
        for model, metrics in efficiency.items()
    ])
    efficiency_df.to_csv('model_efficiency_comparison.csv', index=False, encoding='utf-8-sig')
    print("\nEfficiency metrics saved to 'model_efficiency_comparison.csv'")

    # Save classification reports
    for model_name, metrics in results.items():
        report = metrics['classification_report']
        with open(f'{model_name}_classification_report.txt', 'w') as f:
            f.write(f"{model_name} Classification Report\n")
            f.write(report)
        print(f"Classification report for {model_name} saved to '{model_name}_classification_report.txt'")


    return results, efficiency, loss_history

if __name__ == "__main__":
    main()
