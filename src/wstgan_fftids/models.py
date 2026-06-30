from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFTBandPrior(nn.Module):
    """Learnable FFT band prior replacing the original DWT/wavelet branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        low_cutoff: float = 0.18,
        mid_cutoff: float = 0.36,
    ) -> None:
        super().__init__()
        self.low_cutoff = low_cutoff
        self.mid_cutoff = mid_cutoff
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def _radial_masks(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        yy = torch.fft.fftfreq(height, device=device).view(height, 1)
        xx = torch.fft.fftfreq(width, device=device).view(1, width)
        radius = torch.sqrt(xx * xx + yy * yy)
        low = (radius <= self.low_cutoff).float()
        mid = ((radius > self.low_cutoff) & (radius <= self.mid_cutoff)).float()
        high = (radius > self.mid_cutoff).float()
        full = torch.ones_like(radius)
        return low, mid, high, full

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x_small = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        freq = torch.fft.fft2(x_small, norm="ortho")
        bands = []
        for mask in self._radial_masks(size[0], size[1], x.device):
            mask = mask.view(1, 1, size[0], size[1])
            recon = torch.fft.ifft2(freq * mask, norm="ortho").real
            bands.append(recon)
        band_tensor = torch.cat(bands, dim=1)
        return self.proj(band_tensor)


class TemporalGRUFusion(nn.Module):
    """GRU attention over spatial positions, used as the temporal branch."""

    def __init__(self, channels: int, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        hidden = max(16, int(round(channels * hidden_ratio)))
        self.gru = nn.GRU(channels, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        seq = x.flatten(2).transpose(1, 2)
        out, _ = self.gru(seq)
        out = self.proj(out).transpose(1, 2).reshape(bsz, channels, height, width)
        return out * self.gate(out)


class SpectralSTFusion(nn.Module):
    """Frequency-aware spatio-temporal fusion module."""

    def __init__(self, channels: int, use_temporal: bool = True, temporal_hidden_ratio: float = 0.5) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.temporal = TemporalGRUFusion(channels, hidden_ratio=temporal_hidden_ratio)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, spatial: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(spatial) if self.use_temporal else torch.zeros_like(spatial)
        spectral_gate = spectral.expand_as(spatial) if spectral.shape[-2:] == spatial.shape[-2:] else spectral
        return self.fuse(torch.cat([spatial, temporal, spatial * spectral_gate], dim=1))


class CFFM(nn.Module):
    """Cross-scale feature fusion for decoder skip connections."""

    def __init__(self, in_channels: int, out_channels: int, bottleneck_ratio: float = 1.0) -> None:
        super().__init__()
        hidden_channels = max(8, int(round(out_channels * bottleneck_ratio)))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AblationOptions:
    use_fft_prior: bool = True
    use_temporal: bool = True
    use_st_fusion: bool = True
    use_cffm: bool = True
    fft_low_cutoff: float = 0.18
    fft_mid_cutoff: float = 0.36
    temporal_hidden_ratio: float = 0.5
    cffm_bottleneck_ratio: float = 1.0


class NeutralBandPrior(nn.Module):
    """Unit gate used when the frequency prior branch is ablated."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return x.new_ones((x.size(0), self.out_channels, size[0], size[1]))


class FFTSTGenerator(nn.Module):
    """Encoder-decoder generator with FFT band priors instead of wavelets."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        ablation: AblationOptions | None = None,
    ) -> None:
        super().__init__()
        self.ablation = ablation or AblationOptions()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoders = nn.ModuleList()
        self.priors = nn.ModuleList()
        self.fusions = nn.ModuleList()
        current = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(current, ch, stride=2))
            self.priors.append(
                FFTBandPrior(in_channels, ch, self.ablation.fft_low_cutoff, self.ablation.fft_mid_cutoff)
                if self.ablation.use_fft_prior
                else NeutralBandPrior(ch)
            )
            self.fusions.append(
                SpectralSTFusion(
                    ch,
                    use_temporal=self.ablation.use_temporal,
                    temporal_hidden_ratio=self.ablation.temporal_hidden_ratio,
                )
            )
            current = ch

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1, bias=False)
        self.cffm3 = CFFM(channels[2] * 2, channels[2], self.ablation.cffm_bottleneck_ratio)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1, bias=False)
        self.cffm2 = CFFM(channels[1] * 2, channels[1], self.ablation.cffm_bottleneck_ratio)
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1, bias=False)
        self.cffm1 = CFFM(channels[0] * 2, channels[0], self.ablation.cffm_bottleneck_ratio)
        self.up0 = nn.ConvTranspose2d(channels[0], channels[0], 4, stride=2, padding=1, bias=False)
        self.out = nn.Sequential(
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
            nn.Tanh(),
        )
        self.apply(init_weights)

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = []
        priors = []
        raw = x
        out = x
        for encoder, prior, fusion in zip(self.encoders, self.priors, self.fusions):
            out = encoder(out)
            spectral = prior(raw, out.shape[-2:])
            if self.ablation.use_st_fusion:
                out = fusion(out, spectral)
            features.append(out)
            priors.append(spectral)
        return features, priors

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        feats, priors = self.encode(x)
        x4 = feats[-1]
        d3 = F.relu(self.up3(x4), inplace=True)
        if self.ablation.use_cffm:
            d3 = self.cffm3(torch.cat([d3, feats[2]], dim=1))
        d2 = F.relu(self.up2(d3), inplace=True)
        if self.ablation.use_cffm:
            d2 = self.cffm2(torch.cat([d2, feats[1]], dim=1))
        d1 = F.relu(self.up1(d2), inplace=True)
        if self.ablation.use_cffm:
            d1 = self.cffm1(torch.cat([d1, feats[0]], dim=1))
        d0 = self.up0(d1)
        return self.out(d0), feats, priors

    def frequency_consistency(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        _, real_priors = self.encode(real)
        _, fake_priors = self.encode(fake)
        loss = real.new_tensor(0.0)
        for real_prior, fake_prior in zip(real_priors, fake_priors):
            loss = loss + F.l1_loss(fake_prior, real_prior.detach())
        return loss / len(real_priors)

    def frequency_error(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        _, real_priors = self.encode(real)
        _, fake_priors = self.encode(fake)
        errors = []
        for real_prior, fake_prior in zip(real_priors, fake_priors):
            err = torch.mean(torch.abs(real_prior - fake_prior), dim=(1, 2, 3))
            errors.append(err)
        return torch.stack(errors, dim=1).mean(dim=1)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers = []
        current = in_channels
        for index, ch in enumerate(channels):
            layers.append(nn.Conv2d(current, ch, 4, stride=2, padding=1, bias=False))
            if index > 0:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current = ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(current, 1, 1)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        logits = self.classifier(feat).flatten(1).mean(dim=1)
        return logits, feat


@dataclass
class ModelBundle:
    generator: FFTSTGenerator
    discriminator: Discriminator


def build_models(
    in_channels: int = 3,
    base_channels: int = 32,
    device: torch.device | str = "cpu",
    ablation: AblationOptions | None = None,
) -> ModelBundle:
    generator = FFTSTGenerator(in_channels=in_channels, base_channels=base_channels, ablation=ablation).to(device)
    discriminator = Discriminator(in_channels=in_channels, base_channels=base_channels).to(device)
    return ModelBundle(generator, discriminator)
