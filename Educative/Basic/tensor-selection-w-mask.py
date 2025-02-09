# Selecting tensor with a mask
import torch

matrixT = torch.arange(1, 10).reshape((3,3))

maskBT = torch.BoolTensor([
  [True, False, True],
  [False, False, True],
  [True, False, False]
])

print("The mask tensor is: \n{}".format(maskBT))
print("The original tensor is: \n{}".format(matrixT))
result = torch.masked_select(matrixT, maskBT)
print("The result is {}".format(result))
