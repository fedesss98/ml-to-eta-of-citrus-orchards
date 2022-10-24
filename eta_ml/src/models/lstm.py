import os
import pandas as pd
import sys

import torch

sys.path.append('src')
from data import read_processed_data

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))



data = read_processed_data()
print(data)
y = data.values
print(y.shape)
x = torch.tensor(y)
print(x)
