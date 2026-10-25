"""Per-module rules: which modules carry a constant (bias) term and which break exactness."""

import warnings
from contextlib import contextmanager

import torch
from torch import nn

_BATCHNORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_LINEAR = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
AFFINE = _LINEAR + _BATCHNORM

# Modules that are not piecewise-linear and positively homogeneous. With them the
# decomposition still runs, but scores no longer sum exactly to the output.
# ReLU6/Hardtanh are piecewise-linear but saturate at a constant, which acts as a
# hidden bias that is not accounted for.
_INEXACT = (
    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LocalResponseNorm, nn.GELU, nn.SiLU, nn.Mish, nn.ELU, nn.SELU, nn.CELU,
    nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Softplus, nn.Hardswish, nn.Hardsigmoid,
    nn.ReLU6, nn.Hardtanh, nn.MultiheadAttention,
)


def effective_bias(module: nn.Module) -> torch.Tensor | None:
    """Constant term b of the affine map y = A(x) + b computed by `module` in eval mode.

    For batch norm, y = gamma * (x - mu) / sqrt(var + eps) + beta, so b = beta - gamma * mu / sqrt(var + eps)
    (de la Torre et al., 2020).
    """
    if isinstance(module, _BATCHNORM):
        if module.running_mean is None:
            raise ValueError(f"{module} has track_running_stats=False; eval-mode output is not affine")
        inv_std = torch.rsqrt(module.running_var + module.eps)
        gamma = module.weight if module.affine else torch.ones_like(inv_std)
        beta = module.bias if module.affine else torch.zeros_like(inv_std)
        return beta - gamma * module.running_mean * inv_std
    if isinstance(module, _LINEAR):
        return module.bias
    return None


def check_supported(model: nn.Module) -> list[str]:
    """Warn about modules that break completeness; returns their names."""
    inexact = [name for name, m in model.named_modules() if isinstance(m, _INEXACT)]
    if inexact:
        shown = ", ".join(inexact[:5]) + (" ..." if len(inexact) > 5 else "")
        warnings.warn(
            f"{len(inexact)} module(s) are not piecewise-linear ({shown}); the decomposition "
            "will not sum exactly to the target. Check Decomposition.residual.",
            stacklevel=3,
        )
    return inexact


def bias_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Affine modules of `model` that have a constant term."""
    return {name: m for name, m in model.named_modules()
            if isinstance(m, AFFINE) and effective_bias(m) is not None}


def number_repeats(names: list[str]) -> list[str]:
    """Keys for a sequence of module calls: "name#0", "name#1", ... for a module called more than once."""
    totals = {name: names.count(name) for name in set(names)}
    seen: dict[str, int] = {}
    keys = []
    for name in names:
        keys.append(f"{name}#{seen.get(name, 0)}" if totals[name] > 1 else name)
        seen[name] = seen.get(name, 0) + 1
    return keys


@contextmanager
def recording(layers: dict[str, nn.Module]):
    """Record the outputs of `layers` during a forward pass.

    Yields a list of (key, module, output); a module called several times gets keys
    "name#0", "name#1", ... Each output is cloned before being passed on, so in-place ops
    downstream (ReLU(inplace=True), `out += identity`) cannot overwrite the recorded tensor.
    """
    calls: list[tuple[str, nn.Module, torch.Tensor]] = []

    def record(name):
        def hook(module, inputs, output):
            calls.append((name, module, output))
            return output.clone()
        return hook

    handles = [m.register_forward_hook(record(name)) for name, m in layers.items()]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()
        names = number_repeats([name for name, _, _ in calls])
        calls[:] = [(new, module, output) for new, (_, module, output) in zip(names, calls)]
