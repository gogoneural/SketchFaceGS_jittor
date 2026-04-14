import random
import jittor as jt
from jittor import nn
import numpy as np
from src.models.libs.sketch_model.networks_sketch import define_S
from src.models.libs.sketch_model.create_sketch import SketchSimplifier
from src.models.libs.sketch_model.get_sketch_tradition import random_sketch_extractor

from src.models.libs.sketch_model.sketch_Aug1 import sketch_aug1
from src.models.libs.sketch_model.sketch_aug2 import sketch_aug2


def _jt_normalize(img, mean, std):
    """Normalize a jittor Var image (B,C,H,W) with given mean and std per channel."""
    mean = jt.array(mean).float32().reshape(1, -1, 1, 1)
    std = jt.array(std).float32().reshape(1, -1, 1, 1)
    return (img - mean) / std


class random_get_sketch(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.SketchGen = define_S(3, 3, 64, "sketch_no_part", 'instance', not False, 'normal', 0.02, )
        self.sketch_aug1 = sketch_aug1
        self.sketch_aug2 = sketch_aug2
        self.random_sketch_extractor = random_sketch_extractor
        self.SketchSimplifier = SketchSimplifier(device)

        self.device = device
        for param in self.SketchGen.parameters():
            param.requires_grad = False

    @jt.no_grad()
    def execute(self, img, a=None):
        """Accept jittor Var, return jittor Var."""
        if a is None:
            a = random.random()
        b = random.random()
        normalized = _jt_normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        sketch = ((self.SketchGen(normalized)) + 1) / 2
        if a < 0.2:
            sketch = sketch
        elif 0.2 <= a < 0.4:
            sketch = self.SketchSimplifier(sketch[:, [0]], get_sketch=False)
            sketch = sketch.repeat(1, 3, 1, 1)
        elif 0.4 <= a < 0.5:
            sketch = self.SketchSimplifier(img[:, [0]], get_sketch=True)
            sketch = sketch.repeat(1, 3, 1, 1)
        elif 0.5 <= a < 0.6:
            sketch = self.SketchSimplifier(img[:, [0]], get_sketch=True)
            sketch = self.SketchSimplifier(sketch, get_sketch=False)
            sketch = sketch.repeat(1, 3, 1, 1)
        elif 0.6 <= a < 0.8:
            sketch = self.SketchSimplifier(sketch[:, [0]], get_sketch=False)
            sketch = self.SketchSimplifier(sketch, get_sketch=False)
            sketch = sketch.repeat(1, 3, 1, 1)
        elif 0.8 <= a < 1.0:
            sketch = sketch_aug1(sketch)
            sketch = (sketch + 1) / 2
        sketch = sketch_aug2(sketch.stop_grad())

        return sketch

