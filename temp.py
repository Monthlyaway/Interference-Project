import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class WaveletTransformDemo:
    def __init__(self, scales):
        # Create a simple hyperparameters object to hold our wavelet scales.
        self.hparams = type("HParams", (object,), {})()
        self.hparams.wavelet_scales = scales

        # Create the wavelet filters using the provided scales.
        self.wavelet_filters = self._create_wavelet_bank(scales)

    def _create_wavelet_bank(self, scales):
        max_scale = max(scales)
        # Create a symmetric grid of time points (float for precision)
        t = torch.arange(-max_scale * 2, max_scale *
                         2 + 1, dtype=torch.float32)
        filters = []
        for scale in scales:
            # Construct the wavelet: cosine (carrier) multiplied by a Gaussian window.
            wavelet = torch.cos(1.75 * t / scale) * \
                torch.exp(-(t**2)/(2*(scale**2)))
            # Normalize to preserve energy
            wavelet /= wavelet.norm()
            filters.append(wavelet)
        # Stack filters into a tensor with shape: (num_scales, 1, kernel_length)
        return torch.stack(filters).view(len(scales), 1, -1)

    def _wavelet_transform(self, x: torch.Tensor):
        b, c, l = x.size()
        # Flatten batch and channel dimensions to (B*C, 1, L)
        x_flat = x.view(-1, 1, l)
        # Apply wavelet filters via 1D convolution (padding='same' to preserve length)
        x_conv = F.conv1d(x_flat, self.wavelet_filters, padding='same')
        # Reshape the result back to (B, C, num_scales, L)
        num_scales = len(self.hparams.wavelet_scales)
        return x_conv.view(b, c, num_scales, l)


def plot_wavelet_filters(wavelet_filters, scales):
    # wavelet_filters shape: (num_scales, 1, kernel_length)
    num_scales = wavelet_filters.shape[0]
    kernel_length = wavelet_filters.shape[2]
    # Create a time axis for plotting
    t = np.linspace(-kernel_length//2, kernel_length//2, kernel_length)

    plt.figure(figsize=(12, 4))
    for i in range(num_scales):
        plt.plot(t, wavelet_filters[i, 0, :].numpy(),
                 label=f"Scale {scales[i]}")
    plt.title("Wavelet Filters")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def plot_wavelet_transform_output(x, x_wave, scales):
    # x shape: (B, C, L) and x_wave shape: (B, C, num_scales, L)
    b, c, l = x.shape
    num_scales = x_wave.shape[2]

    # Plot for first sample, first channel
    fig, axes = plt.subplots(
        num_scales + 1, 1, figsize=(12, 2 * (num_scales + 1)))
    t = np.arange(l)

    # Plot original signal
    axes[0].plot(t, x[0, 0, :].numpy(), color='black')
    axes[0].set_title("Original Signal (Sample 0, Channel 0)")

    # Plot wavelet transform output for each scale
    for i in range(num_scales):
        axes[i + 1].plot(t, x_wave[0, 0, i, :].detach().numpy())
        axes[i + 1].set_title(f"Wavelet Transform at Scale {scales[i]}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define wavelet scales (e.g., [4, 8, 16])
    scales = [4, 8, 16]
    demo = WaveletTransformDemo(scales)

    # Plot the wavelet filters
    plot_wavelet_filters(demo.wavelet_filters, scales)

    # Create a dummy input signal consisting of random noise
    B, C, L = 1, 1, 200  # 1 sample, 1 channel, 200 time points
    signal = np.random.randn(L)  # Generate random noise
    x = torch.tensor(signal, dtype=torch.float32).view(B, C, L)

    # Compute the wavelet transform of the signal
    x_wave = demo._wavelet_transform(x)

    # Plot the original random noise signal and the corresponding wavelet transform outputs
    plot_wavelet_transform_output(x, x_wave, scales)
