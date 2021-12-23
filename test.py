import os
import torch
import numpy as np

a = torch.tensor([[1, 2, 1], [3, 4, 3]])
b = torch.tensor([[5, 6, 5], [7, 8, 7]])
c = [a, b]
print(torch.stack(c, 1))