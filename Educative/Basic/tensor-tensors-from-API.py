# Creating tensors from specified APIs
# Joshua Greenfield: I got the tip about aliasing the following as torch.tensor()
# from Meta.ai/c/9dd9413b-4b53-4a11-9617-ae8dc86c050b
# UInt8Tensor
# UInt16Tensor
# UInt32Tensor
# UInt64Tensor

import torch



# BoolTensor False
a = torch.BoolTensor([False])
print(a)

# BoolTensor True
b = torch.BoolTensor([True])
print(b)

# CharTensor 123
c = torch.CharTensor([123])
print(c)

# FloatTensor
d = torch.FloatTensor([1, 2, 3])
print(d)

# IntTensor
e = torch.IntTensor([1, 2, 3])
print(e)

# LongTensor
f = torch.LongTensor([4,5,6])
print(f)

# UInt8Tensor
# Create an alias for torch.tensor() with dtype=torch.uint8 .
torch.UInt8Tensor = lambda *args, **kwargs: torch.tensor(*args, dtype=torch.uint8, **kwargs)
g = torch.UInt8Tensor([123])
print(g)

# UInt16Tensor
# Create an alias for torch.tensor() with dtype=torch.uint16 .
torch.UInt16Tensor = lambda *args, **kwargs: torch.tensor(*args, dtype=torch.uint16, **kwargs)
h = torch.UInt16Tensor([456])
print(h)

# UInt32Tensor
# Create an alias for torch.tensor() with dtype=torch.uint32 .
torch.UInt32Tensor = lambda *args, **kwargs: torch.tensor(*args, dtype=torch.uint32, **kwargs)
i = torch.UInt32Tensor([789])
print(i)

# UInt64Tensor
# Create an alias for torch.tensor() with dtype=torch.uint64 .
torch.UInt64Tensor = lambda *args, **kwargs: torch.tensor(*args, dtype=torch.uint64, **kwargs)
j = torch.UInt64Tensor([1280])
print(j)

# HalfTensor
k = torch.HalfTensor([1.0001])
print(k)

# FloatTensor
l = torch.FloatTensor([6.16])

# DoubleTensor
m = torch.DoubleTensor([9999999999999.9999988888])
