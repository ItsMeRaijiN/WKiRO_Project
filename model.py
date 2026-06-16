import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for groups in [8, 4, 2]:
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.1):
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
    def __init__(self, n_channels: int, latent_dim: int = 24, *, base: int = 32,
                 seq_len: int = 101, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            ConvBlock1d(n_channels, base, stride=2, dropout=dropout),       # T -> T/2
            ConvBlock1d(base, base * 2, stride=2, dropout=dropout),         # -> T/4
            ConvBlock1d(base * 2, base * 4, stride=2, dropout=dropout),     # -> T/8
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, seq_len)
            enc_out = self.encoder(dummy)
        self.enc_channels = enc_out.shape[1]
        self.enc_length = enc_out.shape[2]
        flat = self.enc_channels * self.enc_length

        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, flat),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base * 4, base * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(base * 2), base * 2),
            nn.GELU(),

            nn.ConvTranspose1d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(base), base),
            nn.GELU(),

            nn.ConvTranspose1d(base, base, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(base), base),
            nn.GELU(),

            nn.Conv1d(base, n_channels, kernel_size=1, stride=1),
        )

    @staticmethod
    def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        current = x.shape[-1]
        if current == target_length:
            return x
        if current > target_length:
            return x[..., :target_length]
        return nn.functional.pad(x, (0, target_length - current))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> latent (B, latent_dim)"""
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        return self.to_latent(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_length = x.shape[1]
        x = x.permute(0, 2, 1)                       # (B, C, T)

        z = self.encoder(x)
        latent = self.to_latent(z)                   # (B, latent_dim)

        h = self.from_latent(latent)                 # (B, flat)
        h = h.view(-1, self.enc_channels, self.enc_length)
        out = self.decoder(h)
        out = self._match_length(out, target_length)

        return out.permute(0, 2, 1)                  # (B, T, C)
