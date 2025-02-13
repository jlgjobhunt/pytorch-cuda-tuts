
# Saving and loading tensors
# Be sure to 'mkdir output' in bash before running
# torchrun tensor-saving-loading.py

import torch

# Saving a single tensor

valuable_tensor1 = torch.tensor([1, 2, 3])
torch.save(valuable_tensor1, "./output/tensor.pt")



# Loading tensors from a file.

awakened_tensor1 = torch.load("./output/tensor.pt")

print("The tensor awakened_tensor1 is {}".format(awakened_tensor1))



# Saving multiple tensors.

matrixTi = {}
matrixTi["t1"] = torch.tensor([1, 2, 3])
matrixTi["t2"] = torch.tensor([2, 4, 6])

torch.save(matrixTi, "./output/tensor.pt")

matrixTj = torch.load("./output/tensor.pt")

print("The tensor map is {}".format(matrixTj))

