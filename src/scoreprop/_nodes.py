"""Generic node constants for `decompose(..., nodes="taylor")` (the activation rule of arXiv 1712.08107, extended).

Every differentiable node z = f(u) is replaced, for the given input, by its local linear model
z = J u + c with J = df/du and c = f(u) - J u. Backpropagating through the same J gives
T = <dT/dx, x> + sum_nodes <dT/dz, c> exactly. Affine layers keep their closed-form constant
(`effective_bias`); every other node gets c from one forward-mode product, c = f(u) - jvp(f, u, u).
For ReLU-type units c = 0; for GELU, SiLU, Tanh, ... it is the paper's activation constant
phi(a) - phi'(a) a.

Quantities computed from the image itself are frozen: they are constants of the node for that
image (as batch-norm statistics and dropout masks are in eval mode), so J ignores their dependence
on the input. This covers LayerNorm/GroupNorm statistics, softmax attention weights and
squeeze-excite gates. Forward values are unchanged (straight-through where needed).
"""

import copy
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ._modules import AFFINE, effective_bias, number_repeats

# Nodes whose constant is exactly zero (piecewise-linear and positively homogeneous, or identity).
_ZERO = (
    nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.Identity, nn.Flatten, nn.Unflatten, nn.Upsample, nn.ZeroPad2d,
    nn.modules.dropout._DropoutNd,
    nn.modules.pooling._MaxPoolNd, nn.modules.pooling._AvgPoolNd,
    nn.modules.pooling._AdaptiveMaxPoolNd, nn.modules.pooling._AdaptiveAvgPoolNd,
)
_ZERO_NAMES = ("StochasticDepth", "Permute")
_UNITS = (nn.MultiheadAttention,)  # recorded as one node; their children are not
_PRIMAL_KWARGS = ("query", "key", "value", "input")


@dataclass
class Node:
    key: str
    module: nn.Module
    out: torch.Tensor
    channel_dim: int
    constant: Callable[[], torch.Tensor]
    value: torch.Tensor | None = field(default=None)  # constant, filled in for nodes that reach the target


# --- frozen functionals -------------------------------------------------------------------


def _frozen_layer_norm(original):
    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        dims = tuple(range(-len(normalized_shape), 0))
        mean = input.mean(dims, keepdim=True).detach()
        var = input.var(dims, unbiased=False, keepdim=True).detach()
        frozen = (input - mean) / torch.sqrt(var + eps)
        if weight is not None:
            frozen = frozen * weight
        if bias is not None:
            frozen = frozen + bias
        true = original(input, normalized_shape, weight, bias, eps)
        return true.detach() + (frozen - frozen.detach())
    return layer_norm


def _frozen_group_norm(original):
    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        n, c = input.shape[:2]
        grouped = input.reshape(n, num_groups, -1)
        mean = grouped.mean(-1, keepdim=True).detach()
        var = grouped.var(-1, unbiased=False, keepdim=True).detach()
        frozen = ((grouped - mean) / torch.sqrt(var + eps)).reshape(input.shape)
        shape = [1, c] + [1] * (input.dim() - 2)
        if weight is not None:
            frozen = frozen * weight.view(shape)
        if bias is not None:
            frozen = frozen + bias.view(shape)
        true = original(input, num_groups, weight, bias, eps)
        return true.detach() + (frozen - frozen.detach())
    return group_norm


def _frozen_softmax(original):
    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        return original(input, dim, _stacklevel, dtype).detach()
    return softmax


def _frozen_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Attention with the weights treated as constants: out = w @ v with w = softmax(q k^T) detached."""
    if dropout_p:
        raise ValueError("nodes='taylor' needs eval mode (attention dropout is active)")
    if enable_gqa:
        raise NotImplementedError("grouped-query attention")
    scores = q @ k.transpose(-2, -1) * (scale if scale is not None else 1 / math.sqrt(q.size(-1)))
    if is_causal:
        keep = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
    return torch.softmax(scores, -1).detach() @ v


@contextmanager
def frozen_statistics():
    """Patch torch.nn.functional so image-derived statistics, attention weights and gates are constants."""
    with ExitStack() as stack:
        patches = {"layer_norm": _frozen_layer_norm(F.layer_norm), "group_norm": _frozen_group_norm(F.group_norm),
                   "softmax": _frozen_softmax(F.softmax), "scaled_dot_product_attention": _frozen_attention}
        for name, replacement in patches.items():
            original = getattr(F, name)
            setattr(F, name, replacement)
            stack.callback(setattr, F, name, original)
        try:
            from torchvision.ops.misc import SqueezeExcitation
        except ImportError:  # torchvision is optional
            SqueezeExcitation = None
        if SqueezeExcitation is not None:
            original_forward = SqueezeExcitation.forward
            SqueezeExcitation.forward = lambda self, input: self._scale(input).detach() * input
            stack.callback(setattr, SqueezeExcitation, "forward", original_forward)
        for patch in _timm_gate_patches():  # timm's squeeze-excite blocks (EfficientNet, ResNeSt, ... in timm)
            stack.enter_context(patch)
        yield


def _timm_gate_patches():
    """Context managers that detach the gate of timm's squeeze-excite modules, if timm is installed."""
    try:
        from timm.layers.squeeze_excite import SEModule
        from timm.models._efficientnet_blocks import SqueezeExcite
    except ImportError:  # timm is optional
        return []

    def squeeze_excite(self, x):
        x_se = self.conv_expand(self.act1(self.conv_reduce(x.mean((2, 3), keepdim=True))))
        return x * self.gate(x_se).detach()

    def se_module(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc2(self.act(self.bn(self.fc1(x_se))))
        return x * self.gate(x_se).detach()

    @contextmanager
    def patched(cls, forward):
        original = cls.forward
        cls.forward = forward
        try:
            yield
        finally:
            cls.forward = original

    return [patched(SqueezeExcite, squeeze_excite), patched(SEModule, se_module)]


# --- node constants -----------------------------------------------------------------------


def _is_zero(module: nn.Module) -> bool:
    return isinstance(module, _ZERO) or type(module).__name__ in _ZERO_NAMES


def _nodes_of(model: nn.Module) -> dict[str, nn.Module]:
    """Affine layers, plus every other leaf with a possible constant, plus unit modules."""
    nodes, units = {}, []
    for name, module in model.named_modules():
        if not name or any(name.startswith(u + ".") for u in units):
            continue
        if isinstance(module, _UNITS):
            units.append(name)
            nodes[name] = module
        elif isinstance(module, AFFINE):
            if effective_bias(module) is not None:
                nodes[name] = module
        elif not any(True for _ in module.children()) and not _is_zero(module):
            nodes[name] = module
    return nodes


def channels_last(t: torch.Tensor) -> bool:
    """Whether a 4-D activation of a square image is (B, H, W, C) rather than (B, C, H, W).

    Heuristic on the shape: square positions are dims (1, 2) when dims 1 and 2 are equal and differ from dim 3.
    Wrong only for non-square inputs or C == H == W.
    """
    return t.dim() == 4 and t.shape[1] == t.shape[2] != t.shape[3]


def channel_dim_of(out: torch.Tensor) -> int:
    """Channel dimension of a module output known only by its shape (cuts): last for channels-last and (B, N, D)."""
    if out.dim() == 4:
        return -1 if channels_last(out) else 1
    return -1 if out.dim() in (2, 3) else 1


def _channel_dim(module: nn.Module, out: torch.Tensor) -> int:
    """Dimension that holds the channels or features of `out`; the other dimensions are positions."""
    if isinstance(module, nn.Linear) or type(module) is nn.LayerNorm or isinstance(module, nn.MultiheadAttention):
        return -1
    if out.dim() == 4:
        return -1 if channels_last(out) else 1  # ConvNeXt blocks work on (B, H, W, C)
    if out.dim() == 3 and not isinstance(module, (nn.Conv1d, nn.BatchNorm1d, nn.GroupNorm)):
        return -1  # token sequence (B, N, D)
    return 1


def _taylor_constant(module: nn.Module, args: tuple, kwargs: dict, out: torch.Tensor) -> torch.Tensor:
    """c = f(u) - jvp(f, u, u) for a module called with (args, kwargs) that gave `out`."""
    if getattr(module, "inplace", False):
        module = copy.copy(module)
        module.inplace = False
    where = [("a", i) for i, a in enumerate(args) if torch.is_tensor(a) and a.is_floating_point()]
    where += [("k", k) for k, v in kwargs.items()
              if k in _PRIMAL_KWARGS and torch.is_tensor(v) and v.is_floating_point()]
    primals = tuple(args[i] if kind == "a" else kwargs[i] for kind, i in where)

    def f(*tensors):
        a, kw = list(args), dict(kwargs)
        for (kind, i), t in zip(where, tensors):
            if kind == "a":
                a[i] = t
            else:
                kw[i] = t
        result = module.forward(*a, **kw)
        return result[0] if isinstance(result, tuple) else result

    # grad mode on: keeps nn.MultiheadAttention off its fused fast path, which has no forward-mode AD
    with torch.enable_grad():
        _, jv = torch.func.jvp(f, primals, primals)
    return (out - jv).detach()


def _vit_constant(model: nn.Module) -> Callable[[], torch.Tensor]:
    """torchvision ViT: class token (padded to the sequence) + position embedding enter outside any module."""
    def constant():
        pos = model.encoder.pos_embedding.detach()
        cls = model.class_token.detach().expand(1, -1, -1)
        return pos + torch.cat([cls, torch.zeros_like(pos[:, 1:])], 1)
    return constant


@contextmanager
def taylor_recording(model: nn.Module):
    """Record the node outputs of `model` during a forward pass under frozen statistics.

    Yields a list of `Node`. Call `node.constant()` inside the context (the frozen functionals must
    still be active) for the nodes whose output reaches the target. Outputs are cloned before being
    passed on, as in `recording`.
    """
    records: list[Node] = []
    stash: dict[nn.Module, tuple] = {}
    handles = []
    watch = _nodes_of(model)
    explicit = {}
    if type(model).__name__ == "VisionTransformer" and hasattr(model, "class_token"):
        dropout = model.encoder.dropout
        watch.pop("encoder.dropout", None)
        explicit["encoder.dropout"] = (dropout, _vit_constant(model))

    def pre(module, args, kwargs):
        keep = (lambda t: t.detach().clone()) if getattr(module, "inplace", False) else (lambda t: t.detach())
        stash[module] = (tuple(keep(a) if torch.is_tensor(a) else a for a in args),
                         {k: keep(v) if torch.is_tensor(v) else v for k, v in kwargs.items()})

    def make_post(name, fixed=None):
        def post(module, args, kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            if fixed is not None:
                constant = fixed
            elif isinstance(module, AFFINE):
                def constant(module=module, out=out):
                    bias = effective_bias(module).detach()
                    shape = [1] * out.dim()
                    shape[_channel_dim(module, out)] = -1
                    return bias.view(shape)
            else:
                saved = stash.pop(module)

                def constant(module=module, saved=saved, out=out):
                    return _taylor_constant(module, saved[0], saved[1], out.detach())
            records.append(Node(name, module, out, _channel_dim(module, out), constant))
            clone = out.clone()
            return (clone,) + tuple(output[1:]) if isinstance(output, tuple) else clone
        return post

    for name, module in watch.items():
        if not isinstance(module, AFFINE):
            handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(make_post(name), with_kwargs=True))
    for name, (module, fixed) in explicit.items():
        handles.append(module.register_forward_hook(make_post(name, fixed), with_kwargs=True))
    try:
        with frozen_statistics():
            yield records
    finally:
        for handle in handles:
            handle.remove()
        for node, key in zip(records, number_repeats([n.key for n in records])):
            node.key = key
