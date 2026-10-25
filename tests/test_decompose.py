import pytest
import torch
from torch import nn

from scoreprop import decompose


def relative_residual(dec):
    return (dec.residual.abs() / dec.score.abs().clamp_min(1e-6)).max().item()


@pytest.fixture
def cnn():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4),
    )
    for m in model.modules():  # non-trivial running statistics, so the batch-norm constants are not zero
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2)
            m.bias.data.normal_()
    return model.eval()


def test_exact_on_piecewise_linear_network(cnn):
    dec = decompose(cnn, torch.randn(1, 3, 16, 16), target=[0, 1, 2, 3])
    assert dec.input_map.shape == (4, 16, 16)
    assert relative_residual(dec) < 1e-4
    assert dec.bias_total.abs().max() > 0


def test_reference_keeps_exactness(cnn):
    dec = decompose(cnn, torch.randn(1, 3, 16, 16), target=0, reference=torch.ones(3, 1, 1))
    assert dec.reference_score is not None
    assert relative_residual(dec) < 1e-4


def test_callable_target(cnn):
    dec = decompose(cnn, torch.randn(1, 3, 16, 16), target=lambda out: out[:, 0] - out[:, 1])
    assert relative_residual(dec) < 1e-4


def test_cut_is_complete(cnn):
    dec = decompose(cnn, torch.randn(1, 3, 16, 16), target=0, cuts=["3"])
    assert dec.cut_residual("3").abs().max() < 1e-4


def test_taylor_is_exact_with_smooth_activations():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 8, 3, padding=1), nn.SiLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4),
    ).eval()
    x = torch.randn(1, 3, 16, 16)
    with pytest.warns(UserWarning):
        assert relative_residual(decompose(model, x, target=0)) > 1e-3  # default mode: not exact
    assert relative_residual(decompose(model, x, target=0, nodes="taylor")) < 1e-4


def test_requires_eval_mode(cnn):
    with pytest.raises(ValueError, match="eval"):
        decompose(cnn.train(), torch.randn(1, 3, 16, 16), target=0)
