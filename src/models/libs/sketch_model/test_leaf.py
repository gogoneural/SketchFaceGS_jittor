import torch

ws_sketch = torch.randn(1, 7, 512, requires_grad=True)

print(ws_sketch.is_leaf)
print(list(ws_sketch)[0].is_leaf)


