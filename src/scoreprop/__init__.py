"""Complete score decomposition for piecewise-linear networks (de la Torre et al., arXiv 1712.08107).

The output of `decompose` is the method: the input term S_I on the pixels and one constant term S_k(l) per layer, on that
layer's grid. The constants are not placed on the pixels: inside a linear region they do not depend on the pixel values.
"""

from .decompose import Decomposition, decompose

__version__ = "0.1.0"
__all__ = ["Decomposition", "decompose"]
