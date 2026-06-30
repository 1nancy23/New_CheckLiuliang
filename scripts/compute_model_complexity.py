from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from wstgan_fftids.comparison_models import BiGANModel, FAnoGANModel, MTSDVGANModel, VAEModel
from wstgan_fftids.models import AblationOptions, build_models


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def module_macs(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor | tuple[torch.Tensor, ...]) -> int:
    x = inputs[0]
    y = output[0] if isinstance(output, tuple) else output
    if isinstance(module, nn.Conv2d):
        out = y
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        return int(out.numel() * kernel_ops)
    if isinstance(module, nn.ConvTranspose2d):
        out = y
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.out_channels // module.groups)
        return int(out.numel() * kernel_ops)
    if isinstance(module, nn.Linear):
        return int(x.shape[0] * module.in_features * module.out_features)
    return 0


def count_gflops(forward_fn) -> float:
    macs = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output) -> None:
        nonlocal macs
        macs += module_macs(module, inputs, output)

    modules = []
    for model in forward_fn.models:
        modules.extend(model.modules())
    for module in modules:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        forward_fn()
    for handle in handles:
        handle.remove()
    return macs / 1e9


def main() -> None:
    in_channels = 4
    base_channels = 48
    x = torch.randn(1, in_channels, 16, 16)

    proposed = build_models(in_channels=in_channels, base_channels=base_channels, ablation=AblationOptions())

    vae = VAEModel(base_channels=base_channels, in_channels=in_channels)
    fanogan = FAnoGANModel(base_channels=base_channels, in_channels=in_channels)
    bigan = BiGANModel(base_channels=base_channels, in_channels=in_channels)
    mts = MTSDVGANModel(base_channels=base_channels, in_channels=in_channels)
    for model in (proposed.generator, proposed.discriminator, vae, fanogan, bigan, mts):
        model.eval()

    def proposed_forward() -> None:
        fake = proposed.generator(x)
        if isinstance(fake, tuple):
            fake = fake[0]
        proposed.discriminator(fake)
    proposed_forward.models = [proposed.generator, proposed.discriminator]

    def vae_forward() -> None:
        vae(x)
    vae_forward.models = [vae]

    def fanogan_forward() -> None:
        fake, _ = fanogan(x)
        fanogan.discriminator(x)
        fanogan.discriminator(fake)
    fanogan_forward.models = [fanogan]

    def bigan_forward() -> None:
        fake, z = bigan.reconstruct(x)
        bigan.discriminator(x, z)
    bigan_forward.models = [bigan]

    def mts_forward() -> None:
        fake, *_ = mts(x)
        mts.discriminator(x)
    mts_forward.models = [mts]

    rows = [
        ("IF", 0, 0.0),
        ("VAE", count_params(vae), count_gflops(vae_forward)),
        ("f-AnoGAN", count_params(fanogan), count_gflops(fanogan_forward)),
        ("BiGAN", count_params(bigan), count_gflops(bigan_forward)),
        ("MTS-DVGAN", count_params(mts), count_gflops(mts_forward)),
        ("Proposed", count_params(proposed.generator) + count_params(proposed.discriminator), count_gflops(proposed_forward)),
    ]

    out_path = Path("outputs/model_complexity_latest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "params_m", "gflops"])
        for method, params, gflops in rows:
            writer.writerow([method, f"{params / 1e6:.3f}", f"{gflops:.3f}"])
            print(f"{method}: params={params / 1e6:.3f}M, GFLOPs={gflops:.3f}")
    print(out_path)


if __name__ == "__main__":
    main()
