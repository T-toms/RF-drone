from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectrogramFrontEnd(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 512,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size
        self.register_buffer("window", torch.hamming_window(n_fft), persistent=False)

    def forward(self, iq_components: torch.Tensor) -> torch.Tensor:
        if iq_components.ndim != 3 or iq_components.shape[1] != 2:
            raise ValueError(
                f"Expected input [batch, 2, samples], got {tuple(iq_components.shape)}"
            )
        complex_iq = torch.complex(iq_components[:, 0], iq_components[:, 1])
        spectra = torch.stft(
            complex_iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        log_power = 10.0 * torch.log10(spectra.abs().square() + 1e-10)
        log_power = torch.fft.fftshift(log_power, dim=1)
        mean = log_power.mean(dim=(1, 2), keepdim=True)
        std = log_power.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        normalized = (log_power - mean) / std
        return F.interpolate(
            normalized.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(
        self, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(self.in_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
