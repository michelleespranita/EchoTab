import torch.nn as nn


def build_head(
    in_dim: int,
    num_layers: int,
    dropout: float,
    out_dim: int = 1,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Module:
    """MLP with (num_layers - 1) hidden layers of width in_dim, followed by a linear
    output layer. num_layers=1 is a plain linear head."""
    if num_layers < 1:
        raise ValueError(f"Unsupported number of layers: {num_layers}")
    if num_layers == 1:
        return nn.Linear(in_dim, out_dim)

    layers = []
    for _ in range(num_layers - 1):
        layers += [nn.Linear(in_dim, in_dim), activation(), nn.Dropout(p=dropout)]
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)
