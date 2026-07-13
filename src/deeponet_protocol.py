"""Protocol-conditioned DeepONet modules for V7.

The actual V7 operator is PyTorch-based and must be run in the `pybamm-inv`
conda environment. A small NumPy/sklearn ridge scaffold is retained for cheap
unit tests and non-torch environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


def cycle_fourier_features(cycles: np.ndarray, *, max_cycle: float = 2500.0, basis_dim: int = 32) -> np.ndarray:
    """Encode cycle indices with constant, linear, and Fourier terms."""

    n = np.asarray(cycles, dtype=float).reshape(-1, 1) / float(max_cycle)
    feats = [np.ones_like(n), n]
    n_freq = max(1, (basis_dim - 2) // 2)
    for k in range(1, n_freq + 1):
        feats.append(np.sin(2.0 * np.pi * k * n))
        feats.append(np.cos(2.0 * np.pi * k * n))
    out = np.concatenate(feats, axis=1)
    if out.shape[1] < basis_dim:
        out = np.pad(out, ((0, 0), (0, basis_dim - out.shape[1])))
    return out[:, :basis_dim]


@dataclass
class ProtocolDeepONetRidge:
    """Fixed-feature branch x trunk operator with ridge readout."""

    input_dim: int
    basis_dim: int = 32
    max_cycle: float = 2500.0
    alpha: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state)
        self.branch_weights = rng.normal(0.0, 1.0 / max(1, self.input_dim) ** 0.5, size=(self.input_dim, self.basis_dim))
        self.branch_bias = rng.normal(0.0, 0.05, size=(self.basis_dim,))
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)
        self.fitted = False

    def branch_features(self, x: np.ndarray) -> np.ndarray:
        xs = self.x_scaler.transform(np.asarray(x, dtype=float))
        return np.tanh(xs @ self.branch_weights + self.branch_bias)

    def design_matrix(self, x: np.ndarray, cycles: np.ndarray) -> np.ndarray:
        return self.branch_features(x) * cycle_fourier_features(cycles, max_cycle=self.max_cycle, basis_dim=self.basis_dim)

    def fit(self, x: np.ndarray, cycles: np.ndarray, q: np.ndarray) -> "ProtocolDeepONetRidge":
        x = np.asarray(x, dtype=float)
        q = np.asarray(q, dtype=float).reshape(-1, 1)
        self.x_scaler.fit(x)
        phi = self.design_matrix(x, cycles)
        y = self.y_scaler.fit_transform(q).ravel()
        self.model.fit(phi, y)
        self.fitted = True
        return self

    def predict(self, x: np.ndarray, cycles: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ProtocolDeepONetRidge is not fitted")
        phi = self.design_matrix(x, cycles)
        pred = self.model.predict(phi).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred).ravel()


def torch_positional_encoding(n: "torch.Tensor", pe_dim: int = 8) -> "torch.Tensor":
    """Fourier positional encoding for torch DeepONet trunks."""

    if torch is None:
        raise RuntimeError("PyTorch is required")
    freqs = 2.0 ** torch.arange(pe_dim, dtype=n.dtype, device=n.device)
    args = freqs.unsqueeze(0) * torch.pi * n
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ProtocolBranchNet(nn.Module if nn is not None else object):
    """Branch net for theta/protocol feature vectors."""

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class ProtocolTrunkNet(nn.Module if nn is not None else object):
    """Cycle-index trunk net with Fourier positional encoding."""

    def __init__(self, hidden: int = 32, out_dim: int = 32, pe_dim: int = 8) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.pe_dim = pe_dim
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * pe_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, n: "torch.Tensor") -> "torch.Tensor":
        pe = torch_positional_encoding(n, self.pe_dim)
        return self.net(torch.cat([n, pe], dim=-1))


class ProtocolDeepONet(nn.Module if nn is not None else object):
    """PyTorch protocol-conditioned DeepONet."""

    def __init__(self, branch_in: int, branch_hidden: int = 64, trunk_hidden: int = 32, p: int = 32, pe_dim: int = 8) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.branch = ProtocolBranchNet(branch_in, branch_hidden, p)
        self.trunk = ProtocolTrunkNet(trunk_hidden, p, pe_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: "torch.Tensor", n: "torch.Tensor") -> "torch.Tensor":
        return (self.branch(x) * self.trunk(n)).sum(dim=-1, keepdim=True) + self.bias
