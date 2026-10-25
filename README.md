# scoreprop

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23641811.svg)](https://doi.org/10.5281/zenodo.23641811)

**ScoreProp: receptive-field score decomposition for neural networks.** ScoreProp (de la Torre, Valls & Puig, 2017) splits a network's output score into an input term and one constant term per layer, and maps every term back to the image. For piecewise-linear networks (conv / linear / batch norm, ReLU, pooling, residual additions) the split is exact by construction. With `nodes="taylor"` it also applies to other differentiable networks (MobileNet-V2, EfficientNet-B0, ConvNeXt-T, ViT-B/16), with relative residuals ≤ 3e-6.

The `scoreprop` package is a general PyTorch implementation of the score-propagation method of the Neurocomputing paper and PhD thesis below. It works for any supported PyTorch network.

The software is archived on Zenodo: [doi:10.5281/zenodo.23641811](https://doi.org/10.5281/zenodo.23641811).

## Origin and credit

The method was introduced in:

- J. de la Torre, A. Valls, D. Puig. **A deep learning interpretable classifier for diabetic retinopathy disease grading.** *Neurocomputing* 396 (2020) 465–476. [doi:10.1016/j.neucom.2018.07.102](https://doi.org/10.1016/j.neucom.2018.07.102). Preprint: [arXiv:1712.08107](https://arxiv.org/abs/1712.08107) (21 December 2017).
- J. de la Torre. **Diabetic Retinopathy Classification and Interpretation using Deep Learning Techniques.** PhD thesis, Universitat Rovira i Virgili, Departament d'Enginyeria Informàtica i Matemàtiques, 2019. Supervisors: Aïda Valls Mateu and Domènec Savi Puig Valls. The method is in Chapter 6.

Related work by the same authors, the ICA analysis of the model's features:

- J. de la Torre, A. Valls, D. Puig, P. Romero-Aroca. **Identification and visualization of the underlying independent causes of the diagnostic of diabetic retinopathy made by a deep learning classifier.** [arXiv:1809.08567](https://arxiv.org/abs/1809.08567) (2018).

Author profile: [Google Scholar](https://scholar.google.com/citations?user=tUFCcsgAAAAJ&hl=en).

## The decomposition

For any target T that is linear in the model output (a logit, a difference of logits, a probe on the features):

```
T(x) = sum(S_I) + sum_l sum(S_k(l))      S_I = dT/dx · x,   S_k(l) = dT/dz_l · b_l
```

- **S_I** is the input-dependent score.
- **S_k(l)** is the constant score of layer l (its bias, or the effective bias of batch norm, β − γμ/√(var + ε)). It is kept at that layer's resolution.

All terms come from one forward pass and **one** backward pass. `Decomposition.residual` is the completeness check, ~0 for supported networks.

## Install

```bash
pip install scoreprop
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1. From a clone: `pip install -e ".[dev]"`, then `pytest`. torchvision and timm are optional; when installed, their squeeze-excite blocks are handled by `nodes="taylor"`.

## Quick start

```python
from scoreprop import decompose

dec = decompose(model.eval(), image, target=[0, 1, 2, 3, 4])   # one row per class
dec.input_map                                                   # S_I on the pixels, (classes, H, W)
dec.bias_contributions                                          # S_k(l): each layer's constants, on that layer's grid
dec.residual                                                    # ~0: S_I + sum_l S_k(l) equals the logits
```

### Options

- **Constants stay on their layer's grid.** Inside a linear region the layer constants do not depend on the pixel values, so they have no pixel location, and the library does not project them onto the image. The pixel map is `dec.input_map` (S_I).
- **Reference.** `decompose(..., reference=x_ref)` measures S_I = dT/dx · (x − x_ref) instead of against x = 0, which is the dataset mean colour in normalised space. The scalar dT/dx · x_ref is reported as `dec.reference_score`, and the decomposition stays exact. The default (`reference=None`) is the paper's choice.
- **Intermediate layers.** `dec.layer_scores` holds the score carried by every layer's output (dT/dz_l · z_l), the paper's per-layer maps. They live on each layer's own grid.
- **Any linear target.** `target` can be a callable, e.g. `lambda e: e @ reference` for the similarity of an embedding to a fixed one. Exactness needs the callable to be linear in the model output. Append an affine probe (e.g. an ICA or PCA component of standardised features) to the model as an `nn.Linear`, so its offset appears as one more constant score instead of in the residual.
- **Non-piecewise-linear networks.** `decompose(..., nodes="taylor")` gives every differentiable node its local linear model z = J u + c, with node constant c = f(u) − J u. This generalises the paper's activation rule. Quantities computed from the image itself are frozen as node constants for that image: LayerNorm/GroupNorm statistics, softmax attention weights and squeeze-excite gates. S_I and S_k then describe the network with those quantities fixed. The results are identical to the default mode on ReLU networks. In the default mode, unsupported modules trigger a warning and their constants show up in `dec.residual`.

## Examples

The examples use torchvision models and a normalised image tensor `image` of shape (1, 3, 224, 224).

**Explain two classes of a ResNet and inspect the layer constants.**

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from scoreprop import decompose

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
dec = decompose(model, image, target=[207, 208])      # golden retriever, Labrador

dec.input_map.shape                                   # (2, 224, 224): S_I per pixel, one row per class
dec.residual                                          # ~1e-5: the terms add up to the two logits
for name, s in dec.bias_contributions.items():        # S_k(l) on each layer's grid, e.g. (2, 56, 56)
    print(name, tuple(s.shape), s.flatten(1).sum(1))
```

**Contrast against the other classes** with any target that is linear in the logits:

```python
top = model(image).argmax(1).item()
dec = decompose(model, image, target=lambda out: out[:, top] - out.mean(1))
```

**Measure the input score against a reference image**, here black in normalised space:

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
dec = decompose(model, image, target=207, reference=(0 - mean) / std)
dec.reference_score                                   # dT/dx · x_ref; the residual is still ~0
```

**Get a complete map at an intermediate resolution.** The output of `layer3` is a cut of the graph, so its score plus the constants after it equals the logit:

```python
dec = decompose(model, image, target=207, cuts=["layer3"])
dec.cut_scores["layer3"].shape                        # (1, 14, 14)
dec.cut_residual("layer3")                            # ~0
```

**Per-channel constants and per-layer scores** for concept analyses:

```python
dec = decompose(model, image, target=207, keep_channels=True)
dec.bias_channel_contributions["layer4.2.bn3"].shape  # (1, 2048, 7, 7)
dec.layer_scores["layer4.2.bn3"].shape                # (1, 7, 7): dT/dz · z summed over channels
```

**Transformers and other non-piecewise-linear networks** use the Taylor node rule:

```python
from torchvision.models import vit_b_16, ViT_B_16_Weights

vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
dec = decompose(vit, image, target=207, nodes="taylor")
dec.residual / dec.score                              # relative residual, ~1e-6
dec.input_map                                         # (1, 224, 224)
```

**Plot the input map:**

```python
import matplotlib.pyplot as plt

m = dec.input_map[0]
lim = m.abs().max()
plt.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)     # red raises the score, blue lowers it
plt.colorbar(); plt.axis("off"); plt.show()
```

## Package layout

- `scoreprop/decompose.py`: the decomposition (hooks on every affine layer with a constant term, one `autograd.grad` call).
- `scoreprop/_modules.py`: effective biases, the list of exactness-breaking modules, the recording hooks.
- `scoreprop/_nodes.py`: the Taylor node rule (`nodes="taylor"`) and the frozen statistics.

## Citation

Please refer to the method as **ScoreProp** (receptive-field score decomposition). If you use the method, cite the paper (and the thesis, if relevant). If you use this package, please also cite the software record on Zenodo ([doi:10.5281/zenodo.23641811](https://doi.org/10.5281/zenodo.23641811)).

The method:

```bibtex
@article{delatorre2020interpretable,
  title   = {A deep learning interpretable classifier for diabetic retinopathy disease grading},
  author  = {de la Torre, Jordi and Valls, Aida and Puig, Domenec},
  journal = {Neurocomputing},
  volume  = {396},
  pages   = {465--476},
  year    = {2020},
  doi     = {10.1016/j.neucom.2018.07.102},
  note    = {Preprint arXiv:1712.08107, December 2017}
}

@phdthesis{delatorre2019thesis,
  title  = {Diabetic Retinopathy Classification and Interpretation using Deep Learning Techniques},
  author = {de la Torre, Jordi},
  school = {Universitat Rovira i Virgili},
  year   = {2019}
}
```

The software:

```bibtex
@software{delatorre2026scoreprop,
  title     = {ScoreProp: receptive-field score decomposition for neural networks},
  author    = {de la Torre, Jordi},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23641811},
  url       = {https://doi.org/10.5281/zenodo.23641811}
}
```

The ICA analysis:

```bibtex
@misc{delatorre2018ica,
  title         = {Identification and Visualization of the Underlying Independent Causes of the Diagnostic of
                   Diabetic Retinopathy made by a Deep Learning Classifier},
  author        = {de la Torre, Jordi and Valls, Aida and Puig, Domenec and Romero-Aroca, Pere},
  year          = {2018},
  eprint        = {1809.08567},
  archivePrefix = {arXiv}
}
```
