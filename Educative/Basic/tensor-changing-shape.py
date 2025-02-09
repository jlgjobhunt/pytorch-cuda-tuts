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
print("The original shape of vectorT1 is {}. \n".format(vectorT1.shape))
print("The original vectorT1 tensor is {}. \n".format(vectorT1))

vectorT2 = torch.squeeze(vectorT1, dim=1)
print("The new shape of vectorT2 is {}. \n".format(vectorT2.shape))
print("The new tensor is {}. \n".format(vectorT2))

vectorT3 = torch.ones((3, 1, 2, 1, 2))
print("The original shape of vectorT3 is {}. \n".format(vectorT3.shape))


# Un-squeezing a Tensor

vectorT4 = torch.ones((3,3))
print("The original shape of vectorT4 is {}.".format(vectorT4.shape))
print("The original vectorT4 tensor is {}.".format(vectorT4))

vectorT5 = torch.unsqueeze(vectorT4, dim=1)
print("The shape of vectorT4 has been mutated in vectorT5 to be {}.".format(vectorT5.shape))
print("The new tensor is {}.".format(vectorT5))
