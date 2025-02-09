# Selecting tensor with index_select

import torch

# Create a tensor 
a = torch.arange(1, 10).reshape((3, 3))
print(a)
print("The dtype for a is {}. \n".format(a.dtype))

indices = torch.LongTensor([0, 2])

result = torch.index_select(a, dim=0, index=indices)
print(result)
