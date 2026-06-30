from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64, variational: bool = False, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.variational = variational
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = base_channels * 4
        self.fc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim) if variational else None
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        feat = self.features(x).flatten(1)
        if not self.variational:
            return self.fc(feat)
        mu = self.fc(feat)
        assert self.fc_logvar is not None
        logvar = self.fc_logvar(feat)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), -1, 1, 1)
        return self.net(x)


class ImageDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(base_channels * 4, 1)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x).flatten(1)
        return self.classifier(feat).squeeze(1), feat


class JointDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.image = ImageDiscriminator(base_channels, in_channels=in_channels)
        self.joint = nn.Sequential(
            nn.Linear(base_channels * 4 + latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, feat = self.image(x)
        joint_feat = torch.cat([feat, z], dim=1)
        return self.joint(joint_feat).squeeze(1), joint_feat


class VAEModel(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, variational=True, base_channels=base_channels, in_channels=in_channels)
        self.decoder = ConvDecoder(latent_dim, base_channels=base_channels, out_channels=in_channels)

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        return self.decoder(self.sample(mu, logvar)), mu, logvar


class FAnoGANModel(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, variational=False, base_channels=base_channels, in_channels=in_channels)
        self.decoder = ConvDecoder(latent_dim, base_channels=base_channels, out_channels=in_channels)
        self.discriminator = ImageDiscriminator(base_channels, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class BiGANModel(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, variational=False, base_channels=base_channels, in_channels=in_channels)
        self.generator = ConvDecoder(latent_dim, base_channels=base_channels, out_channels=in_channels)
        self.discriminator = JointDiscriminator(latent_dim, base_channels, in_channels=in_channels)
        self.latent_dim = latent_dim

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.generator(z), z


class MTSDVGANModel(nn.Module):
    """A compact dual-variational GAN baseline for image-encoded traffic windows."""

    def __init__(self, latent_dim: int = 64, base_channels: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        branch_dim = latent_dim // 2
        self.local_encoder = ConvEncoder(branch_dim, variational=True, base_channels=base_channels, in_channels=in_channels)
        self.global_encoder = ConvEncoder(branch_dim, variational=True, base_channels=base_channels, in_channels=in_channels)
        self.decoder = ConvDecoder(latent_dim, base_channels=base_channels, out_channels=in_channels)
        self.discriminator = ImageDiscriminator(base_channels, in_channels=in_channels)

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor):
        low = F.avg_pool2d(x, 2)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mu_l, log_l = self.local_encoder(x - low)
        mu_g, log_g = self.global_encoder(low)
        z = torch.cat([self.sample(mu_l, log_l), self.sample(mu_g, log_g)], dim=1)
        return self.decoder(z), mu_l, log_l, mu_g, log_g


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
