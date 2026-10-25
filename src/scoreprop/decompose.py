"""Exact score decomposition: target = sum(S_I) + sum_l sum(S_k(l)).

For a network built from affine layers (conv, linear, batch norm in eval mode) and
piecewise-linear homogeneous operations (ReLU, max/avg pooling, residual addition), the
target is, for the activation pattern of a given input x,

    T(x) = <dT/dx, x> + sum_l <dT/dz_l, b_l>

where z_l is the output of affine layer l and b_l its constant term. The first term is the
input-dependent score S_I of de la Torre et al. (2020, arXiv 1712.08107); each term of the sum is the constant score
S_k of layer l, kept at the resolution of z_l. The score carried by z_l itself, dT/dz_l * z_l,
gives the intermediate (per-layer) maps; in a chain network it sums to T minus the
constant scores of the layers after l. All terms come from a single backward pass.

With a reference input x_ref, S_I = dT/dx * (x - x_ref) and the scalar <dT/dx, x_ref> (the
value of the local linear model at x_ref) is reported separately as `reference_score`. In
normalised space x = 0 is the dataset mean colour; a reference such as the black/masked
image puts zero input score wherever x equals it, e.g. on a masked background.
"""

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field

import torch
from torch import nn

from ._modules import bias_layers, check_supported, effective_bias, recording
from . import _nodes
from ._nodes import Node, taylor_recording

Target = int | Sequence[int] | Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Decomposition:
    """Result of `decompose`. Leading dimension B indexes explained targets."""

    score: torch.Tensor                          # (B,) target values T(x)
    input_contribution: torch.Tensor             # (B, C, *spatial) S_I per input channel
    bias_contributions: dict[str, torch.Tensor]  # name -> (B, *spatial_l); (B,) for linear layers
    input_shape: tuple[int, ...]                 # spatial shape of the input
    # name -> score carried by the layer output, dT/dz_l * z_l summed over channels; same shapes
    layer_scores: dict[str, torch.Tensor] = field(default_factory=dict)
    # name -> per-channel constant scores, (B, C_l, *spatial_l); (B, F) for linear layers
    bias_channel_contributions: dict[str, torch.Tensor] | None = None
    inexact_modules: list[str] = field(default_factory=list)
    # (B,) <dT/dx, x_ref> when a reference input was given, else None
    reference_score: torch.Tensor | None = None
    nodes: str = "exact"  # node rule that produced it
    # cut name -> dT/dz * z at that module's output, summed over channels, (B, *spatial_cut)
    cut_scores: dict[str, torch.Tensor] = field(default_factory=dict)
    # cut name -> keys of the constant terms computed after the cut (downstream of it)
    cut_after: dict[str, list[str]] = field(default_factory=dict)

    def cut_residual(self, cut: str) -> torch.Tensor:
        """T(x) - (score at the cut + constants after it): ~0 when the module output is a complete cut
        of the graph (every path from the input to the output passes through it)."""
        after = sum((self.bias_contributions[k].reshape(len(self.score), -1).sum(1) for k in self.cut_after[cut]),
                    torch.zeros_like(self.score))
        return self.score - self.cut_scores[cut].reshape(len(self.score), -1).sum(1) - after

    @property
    def input_map(self) -> torch.Tensor:
        """S_I summed over input channels, (B, *spatial)."""
        return self.input_contribution.sum(1)

    @property
    def bias_total(self) -> torch.Tensor:
        """Sum over all layers of the constant scores, (B,)."""
        total = torch.zeros_like(self.score)
        for contribution in self.bias_contributions.values():
            total = total + contribution.reshape(len(total), -1).sum(1)
        return total

    @property
    def residual(self) -> torch.Tensor:
        """T(x) - (sum S_I + sum S_k + reference score); ~0 for supported networks."""
        residual = self.score - self.input_map.flatten(1).sum(1) - self.bias_total
        return residual if self.reference_score is None else residual - self.reference_score


def _resolve_target(target: Target, x: torch.Tensor) -> tuple[torch.Tensor, Callable]:
    """Turn class indices into a score function; replicate a single input once per class."""
    if callable(target):
        return x, target
    if isinstance(target, int):
        classes = torch.full((x.shape[0],), target, dtype=torch.long)
    else:
        classes = torch.as_tensor(list(target), dtype=torch.long)
        if x.shape[0] == 1:
            x = x.repeat(len(classes), *([1] * (x.dim() - 1)))
        elif len(classes) != x.shape[0]:
            raise ValueError("pass one class per input, or a single input with several classes")
    classes = classes.to(x.device)

    def class_score(output: torch.Tensor) -> torch.Tensor:
        return output.gather(1, classes[:, None]).squeeze(1)

    return x, class_score


def decompose(
    model: nn.Module,
    x: torch.Tensor,
    target: Target,
    *,
    keep_channels: bool = False,
    reference: torch.Tensor | None = None,
    nodes: str = "exact",
    cuts: Sequence[str] = (),
) -> Decomposition:
    """Decompose `target` of `model(x)` into input and per-layer constant contributions.

    Args:
        model: network in eval mode.
        x: input batch (B, C, *spatial).
        target: class index (same for every input), a sequence of class indices (one per
            input, or several for a single input), or a callable mapping the model output
            to one scalar per input, e.g. a linear probe or a dot product with a fixed
            embedding. Exactness requires the callable to be linear in the model output;
            any constant offset in it shows up in `residual`.
        keep_channels: also return per-channel constant scores (needed for concept analyses).
        reference: input broadcastable to x (e.g. (C, 1, 1) for a constant colour). S_I is
            then dT/dx * (x - reference) and <dT/dx, reference> goes to `reference_score`.
        nodes: "exact" (default) handles affine layers and ReLU-type units and warns about other
            modules. "taylor" gives every differentiable node its local linear model (constant
            c = f(u) - J u, the paper's activation rule), so GELU, SiLU, ReLU6, ... are exact.
            Statistics computed from the image (LayerNorm, GroupNorm), attention weights and
            squeeze-excite gates are frozen, i.e. treated as constants of the node for that image.
        cuts: names of modules whose output is a complete cut of the graph (e.g. "layer3" of a ResNet,
            "features.33" of a sequential model). For each, `cut_scores` holds the score carried by the
            output and `cut_after` the constant terms downstream, so that T = sum(cut score) + sum(those
            constants) (`cut_residual`): a complete map at the resolution of the cut.
    """
    if nodes not in ("exact", "taylor"):
        raise ValueError("nodes must be 'exact' or 'taylor'")
    if model.training:
        raise ValueError("call model.eval() first: batch norm must use running statistics")
    taylor = nodes == "taylor"
    inexact = [] if taylor else check_supported(model)
    x, score_fn = _resolve_target(target, x)
    x = x.detach().requires_grad_(True)

    cut_outputs: dict[str, tuple[torch.Tensor, int]] = {}  # name -> (output, number of nodes recorded before it)
    with ExitStack() as stack:
        if taylor:
            records = stack.enter_context(taylor_recording(model))
        else:
            calls = stack.enter_context(recording(bias_layers(model)))
        for name in cuts:
            def cut_hook(module, inputs, output, name=name):
                if name in cut_outputs:
                    raise ValueError(f"cut {name!r} is called more than once")
                cut_outputs[name] = (output, len(records if taylor else calls))
                return output.clone()  # as in `recording`: protect the recorded tensor from in-place ops
            stack.callback(model.get_submodule(name).register_forward_hook(cut_hook).remove)
        stack.enter_context(torch.enable_grad())
        score = score_fn(model(x))
        if score.shape != (x.shape[0],):
            raise ValueError(f"target must give one scalar per input, got shape {tuple(score.shape)}")
        outputs = [n.out for n in records] if taylor else [out for _, _, out in calls]
        cut_tensors = [cut_outputs[name][0] for name in cuts]
        grads = torch.autograd.grad(score.sum(), [x] + outputs + cut_tensors, allow_unused=True)
        cut_grads = grads[len(grads) - len(cut_tensors):]
        grads = grads[:len(grads) - len(cut_tensors)]
        if taylor:  # constants are computed while the frozen functionals are still active
            for node, grad in zip(records, grads[1:]):
                if grad is not None:
                    node.value = node.constant().detach()

    if not taylor:  # (key, output, channel dim, constant) per recorded affine layer
        records = []
        for key, module, out in calls:
            channel_dim = -1 if isinstance(module, nn.Linear) else 1  # Linear: last dim; conv, BN: dim 1
            records.append(Node(key, module, out, channel_dim, None, effective_bias(module).detach()))
    bias_maps: dict[str, torch.Tensor] = {}
    layer_scores: dict[str, torch.Tensor] = {}
    bias_channels: dict[str, torch.Tensor] = {}
    for node, grad in zip(records, grads[1:]):
        if grad is None:  # module output does not reach the target
            continue
        constant = node.value
        if not taylor:
            shape = [1] * grad.dim()
            shape[node.channel_dim] = -1
            constant = constant.view(shape)
        per_channel = grad.detach() * constant
        bias_maps[node.key] = per_channel.sum(node.channel_dim)
        layer_scores[node.key] = (grad * node.out).detach().sum(node.channel_dim)
        if keep_channels:
            bias_channels[node.key] = per_channel

    keys = [n.key for n in records]
    cut_scores, cut_after = {}, {}
    for name, grad in zip(cuts, cut_grads):
        out, before = cut_outputs[name]
        cut_scores[name] = (grad * out).detach().sum(_nodes.channel_dim_of(out))
        cut_after[name] = [k for k in keys[before:] if k in bias_maps]

    grad_x = grads[0].detach()
    reference_score = None
    if reference is not None:
        reference = torch.as_tensor(reference, dtype=x.dtype, device=x.device).expand_as(x)
        reference_score = (grad_x * reference).flatten(1).sum(1)
    delta = x.detach() if reference is None else x.detach() - reference
    return Decomposition(
        score=score.detach(),
        input_contribution=grad_x * delta,
        bias_contributions=bias_maps,
        input_shape=tuple(x.shape[2:]),
        layer_scores=layer_scores,
        bias_channel_contributions=bias_channels if keep_channels else None,
        inexact_modules=inexact,
        reference_score=reference_score,
        nodes=nodes,
        cut_scores=cut_scores,
        cut_after=cut_after,
    )
