import os
import torch
import numpy as np

a = torch.arange(1, 5).expand(2, -1)
print(a)