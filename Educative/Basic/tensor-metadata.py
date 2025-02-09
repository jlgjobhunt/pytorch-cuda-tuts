#### Tensor Metadata

# Importing PyTorch for all examples.
import torch

###### Getting type from dtype
# import torch
a = torch.tensor([1, 2, 3], dtype=torch.float)
print(a)
print(a.dtype)


###### Getting the number of dim

# import torch

a = torch.ones((3, 4, 6))
print(a)
print(a.ndim)
print(a.dim())


###### Getting the number of elements
# import torch


a = torch.ones((3, 4, 6))
print(a)
print(a.numel())


###### Getting the device 
###### & whether tensor is on CPU or GPU.
# import torch

a = torch.randn((2, 3, 4), dtype=torch.float)

print("The dtype of tensor a is {}. \n".format(a.dtype))

print("The size of tensor a is {}.".format(a.size()))
print("The shape of tensor a is {}. \n".format(a.shape))

print("The dims of tensor a is {}.".format(a.dim()))
print("The dims of tensor a is {}. \n".format(a.ndim))
