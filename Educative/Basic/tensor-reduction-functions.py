# Reduction functions

import torch

vectorT12 = torch.randn((3, 4))
print("The original tensor is: \n {}".format(vectorT12))

print("="*30)
vectorT13 = torch.mean(vectorT12, dim=1)
print("The mean value of dim=1 \n {}".format(vectorT13))

print("="*30)
vectorT14 = torch.sum(vectorT12, dim=0)
print("The sum value of dim=0 \n {}".format(vectorT14))


