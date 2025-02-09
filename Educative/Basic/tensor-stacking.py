# Stacking the Tensors
import torch

vectorT7 = torch.randn((2, 2))
vectorT8 = torch.randn((2, 2))

print("The original tensor vectorT7 is: \n {}:".format(vectorT7))
print(vectorT7)

print("The original tensor vectorT8 is: \n {}:".format(vectorT8))
print(vectorT8)

resultStack = torch.stack((vectorT7, vectorT8), dim=1)

print("The shape of result is {}.".format(resultStack.shape))

print("The new tensor is: \n {}".format(resultStack))
print(resultStack)