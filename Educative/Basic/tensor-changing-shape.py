# Changing the Shape of a Tensor


import torch



# Reshaping a Tensor
matrixTa = torch.arange(1, 9)
print("The original tensor. \n")
print(matrixTa)

matrixTb = torch.reshape(matrixTa, (2, 4))
print("The reshape tensor with shape (2, 4) \n")
print(matrixTb)

matrixTc = torch.reshape(matrixTa, (2, -1))
print("The reshape tensor with shape (2, -1) \n")


# Squeezing a Tensor
vectorT1 = torch.ones((3, 1, 2))
print("The original shape of vectorT1 is {}".format(vectorT1.shape))
print("The original vectorT1 tensor is {}".format(vectorT1))

vectorT2 = torch.squeeze(vectorT1, dim=1)
print("The new shape of vectorT2 is {}".format(vectorT2.shape))
print("The new tensor is {}".format(vectorT2))

vectorT3 = torch.ones((3, 1, 2, 1, 2))
print("The original shape of vectorT3 is {}.".format(vectorT3.shape))

