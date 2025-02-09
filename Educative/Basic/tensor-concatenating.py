# Concatenating Tensors

import torch

matrixTd = torch.randn((3, 3))

print("The original tensor matrixTd is \n {}".format(matrixTd))
resultTd = torch.cat((matrixTd, matrixTd), dim=0)
print(resultTd)

print("The shape of result is {}".format(resultTd.shape))
print("The new tensor is \n {}".format(resultTd))
print(resultTd)


matrixTe = torch.randn((4, 4))
print("The original tensor matrixTe is \n {}".format(matrixTe))
resultTe = torch.cat((matrixTe, matrixTe), dim=1)
print("The shape of result is {}".format(resultTe.shape))
print(resultTe)
