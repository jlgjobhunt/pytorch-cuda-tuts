###### Creating a tensor from a NumPy array

import torch
import numpy as np

na = np.array([1, 2, 3])

# Create a tensor from a NumPy array.
a = torch.tensor(na)

# Create a tensor by from_numpy function.
b = torch.from_numpy(na)


print(a)

print(b)

