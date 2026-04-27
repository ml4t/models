"""Neural components for supervised autoencoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GaussianNoise(nn.Module):
    """Additive Gaussian noise applied during training."""

    def __init__(self, std: float = 0.1) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class SupervisedAutoencoder(nn.Module):
    """Three-head supervised autoencoder."""

    bn_eps = 1e-3
    bn_momentum = 0.01

    def __init__(
        self,
        n_features: int,
        n_labels: int = 1,
        hidden_units: tuple[int, ...] | None = None,
        dropout_rates: tuple[float, ...] | None = None,
        noise_std: float = 0.035,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        if hidden_units is None:
            hidden_units = (96, 96, 896, 448, 448, 256)
        if dropout_rates is None:
            dropout_rates = (0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.272, 0.438)
        if len(hidden_units) != 6:
            raise ValueError(f"hidden_units must contain 6 entries; got {len(hidden_units)}")
        if len(dropout_rates) != 8:
            raise ValueError(f"dropout_rates must contain 8 entries; got {len(dropout_rates)}")
        if output_activation not in {"sigmoid", "linear", "identity"}:
            raise ValueError("output_activation must be one of {'sigmoid', 'linear', 'identity'}")

        self.input_bn = nn.BatchNorm1d(n_features, eps=self.bn_eps, momentum=self.bn_momentum)
        self.input_noise = GaussianNoise(noise_std)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0], eps=self.bn_eps, momentum=self.bn_momentum),
            Swish(),
        )

        self.decoder_dropout = nn.Dropout(dropout_rates[1])
        self.decoder = nn.Linear(hidden_units[0], n_features)

        output_head = nn.Sigmoid() if output_activation == "sigmoid" else nn.Identity()
        self.aux_head = nn.Sequential(
            nn.Linear(n_features, hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1], eps=self.bn_eps, momentum=self.bn_momentum),
            Swish(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(hidden_units[1], n_labels),
            output_head,
        )

        concat_dim = n_features + hidden_units[0]
        self.main_bn = nn.BatchNorm1d(concat_dim, eps=self.bn_eps, momentum=self.bn_momentum)
        self.main_dropout_input = nn.Dropout(dropout_rates[3])

        layers: list[nn.Module] = []
        in_dim = concat_dim
        for i, out_dim in enumerate(hidden_units[2:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim, eps=self.bn_eps, momentum=self.bn_momentum),
                    Swish(),
                    nn.Dropout(dropout_rates[min(i + 4, len(dropout_rates) - 1)]),
                ]
            )
            in_dim = out_dim
        self.main_mlp = nn.Sequential(*layers)
        self.main_output = nn.Sequential(nn.Linear(in_dim, n_labels), output_head)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.input_bn(x)
        x_noisy = self.input_noise(x_norm)
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(self.decoder_dropout(encoded))
        aux_pred = self.aux_head(decoded)

        concat = torch.cat([x_norm, encoded], dim=1)
        concat = self.main_bn(concat)
        concat = self.main_dropout_input(concat)
        main_pred = self.main_output(self.main_mlp(concat))
        return decoded, aux_pred, main_pred

    def get_betas(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_bn(x)
        return self.encoder(x_norm)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, _, main_pred = self.forward(x)
            return main_pred
        finally:
            self.train(was_training)
