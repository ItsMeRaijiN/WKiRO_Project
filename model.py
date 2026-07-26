import torch
from torch import nn

def _group_count(channels: int) -> int:
    for groups in [8, 4, 2]:
        if channels % groups == 0:
            return groups
    return 1

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        x = self.block(x)
        return self.activation(x + residual)

class Conv1dCAE(nn.Module):
    def __init__(self, n_channels: int, latent_channels: int = 128):
        super().__init__()
        if n_channels < 1 or latent_channels < 1:
            raise ValueError("n_channels and latent_channels must be positive.")
        self.n_channels = n_channels

        self.encoder = nn.Sequential(
            ConvBlock1d(n_channels, 64, stride=2),
            ConvBlock1d(64, 128, stride=2),
            ConvBlock1d(128, 128, stride=1, dilation=2),
            ConvBlock1d(128, latent_channels, stride=1, dilation=4),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(128), 128),
            nn.GELU(),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(64), 64),
            nn.GELU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_count(64), 64),
            nn.GELU(),

            nn.Conv1d(64, n_channels, kernel_size=1, stride=1),
        )

    @staticmethod
    def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = x.shape[-1]
        if current_length == target_length:
            return x
        if current_length > target_length:
            return x[..., :target_length]

        pad_amount = target_length - current_length
        return nn.functional.pad(x, (0, pad_amount))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, time, channels).")
        if x.shape[-1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, received {x.shape[-1]}."
            )
        target_length = x.shape[1]

        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)

        z = self.encoder(x)
        out = self.decoder(z)
        out = self._match_length(out, target_length)

        return out.permute(0, 2, 1)
