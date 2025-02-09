# Element-wise Mathematical Operations on Tensors
import torch


# Math operations with scalar
tensorA = torch.tensor([1, 2, 3])
print(tensorA)

resultA = tensorA + 3
print(resultA)

resultA = tensorA.add(3)
print(resultA)

