# Math operation between tensors
import torch

vectorT9 = torch.tensor([1, 2, 3])
vectorT10 = torch.tensor([2, 4, 6])

vectorT11 = vectorT9 + vectorT10

print("The multiple between vectorT9 and vectorT10 is {}.".format(vectorT11))

vectorT11 = vectorT9.mul(vectorT10)
print("The multiple between vectorT9 and vectorT10 is {}.".format(vectorT11))

matrixTe = torch.tensor([[1, 2], [3, 4]])

