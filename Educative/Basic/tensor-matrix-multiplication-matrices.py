# Matrix multiplication with matrices

import torch


matrixTg = torch.ones((2, 4))
matrixTh = torch.ones((4, 3))

resultC = torch.mm(matrixTg, matrixTh)
print("The result is: \n {}".format(resultC))

print("="*30)
resultC = matrixTg.mm(matrixTh)
print("The result is \n {}".format(resultC))


