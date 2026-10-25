"""Decompose ResNet-50 and ViT-B/16 scores on a random image (replace `image` with a normalised photo)."""

import torch
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

from scoreprop import decompose

image = torch.randn(1, 3, 224, 224)

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
dec = decompose(model, image, target=[207, 208], cuts=["layer3"])
print("input map", tuple(dec.input_map.shape), "residual", dec.residual)
for name, s in list(dec.bias_contributions.items())[:3]:
    print(name, tuple(s.shape))
print("layer3 cut", tuple(dec.cut_scores["layer3"].shape), dec.cut_residual("layer3"))

vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
dec = decompose(vit, image, target=207, nodes="taylor")
print("ViT relative residual", (dec.residual / dec.score).item())
