import numpy as np
import pytest

from src.deeponet_protocol import ProtocolDeepONet, ProtocolDeepONetRidge, cycle_fourier_features, torch


def test_cycle_fourier_features_shape():
    feats = cycle_fourier_features(np.array([1, 10, 100]), basis_dim=16)
    assert feats.shape == (3, 16)


def test_protocol_deeponet_ridge_smoke_fit_predict():
    x = np.tile(np.array([[0.9, 0.5, 0.55]]), (20, 1))
    cycles = np.linspace(1, 200, 20)
    q = 1.1 - 0.0002 * cycles
    model = ProtocolDeepONetRidge(input_dim=3, basis_dim=16, alpha=0.1)
    model.fit(x, cycles, q)
    pred = model.predict(x, cycles)
    assert pred.shape == q.shape
    assert np.mean(np.abs(pred - q)) < 0.02


@pytest.mark.skipif(torch is None, reason="PyTorch not installed in active env")
def test_protocol_deeponet_torch_forward():
    model = ProtocolDeepONet(branch_in=4)
    x = torch.randn(5, 4)
    n = torch.rand(5, 1)
    y = model(x, n)
    assert tuple(y.shape) == (5, 1)
