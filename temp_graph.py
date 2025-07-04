import torch
import torch.nn.functional as F
import numpy as np
import proplot as pplt

# --- WaveletVisualizer class remains the same ---
class WaveletVisualizer:
    def __init__(self, scales):
        self.scales = scales
        self.filters = self._create_filterbank(scales)

    def _create_filterbank(self, scales):
        max_scale = max(scales)
        t = torch.arange(-max_scale * 4, max_scale * 4 + 1, dtype=torch.float32)
        bank = []
        for s in scales:
            wavelet = torch.cos(1.75 * t / s) * torch.exp(-t**2 / (2 * s**2))
            wavelet /= torch.linalg.norm(wavelet)
            bank.append(wavelet)
        return torch.stack(bank).view(len(scales), 1, -1)

    def _transform(self, x):
        x_3d = x.view(-1, 1, x.size(-1))
        x_conv = F.conv1d(x_3d, self.filters, padding='same')
        return x_conv.view(*x.shape[:-1], len(self.scales), x.size(-1))

# 1. --- Data Preparation ---
scales = [4, 8, 16]
visualizer = WaveletVisualizer(scales)
torch.manual_seed(0)
signal = torch.randn(1, 1, 200)
transformed = visualizer._transform(signal)

# 2. --- Plotting Setup ---
# Define a clean 2x2 layout
fig, axs = pplt.subplots(nrows=2, ncols=2, figsize=(7, 5.5), sharex=False, sharey=False)

# CORRECT WAY to get a list of colors that can be indexed.
colors = pplt.get_colors('colorblind', N=len(scales))
handles = [] # For the figure-wide legend

# 3. --- Populate Subplots ---

# (a) Top-Left: Wavelet Kernels
ax = axs[0]
for i, s in enumerate(scales):
    kernel = visualizer.filters[i, 0].detach().numpy()
    # Manually assign color from our list
    h = ax.plot(kernel, color=colors[i], label=f'Scale $s={s}$')
    handles.append(h[0])
ax.format(title='Wavelet Kernels')

# (b) Top-Right: Input Signal
ax = axs[1]
ax.plot(signal[0, 0].numpy(), color='k')
ax.format(title='Input Signal')

# (c) Bottom-Left: Response for the first scale
ax = axs[2]
ax.plot(transformed[0, 0, 0].detach().numpy(), color=colors[0])
ax.format(title=f'Response ($s={scales[0]}$)')

# (d) Bottom-Right: Response for the second scale
ax = axs[3]
ax.plot(transformed[0, 0, 1].detach().numpy(), color=colors[1])
ax.format(title=f'Response ($s={scales[1]}$)')


# 4. --- Final Formatting ---
axs.format(
    suptitle='Wavelet Transform Analysis',
    abc=True, # Add subplot labels (a), (b), (c)...
    abcloc='upper left',
    xlabel='Time (samples)',
    ylabel='Amplitude'
)

# Add a single, clean legend to the right of the figure
fig.legend(handles, loc='r', title='Kernel Scale')

fig.save('doc/images/wavelet-transform-subplots-corrected.pdf', dpi=600)
# fig.show()