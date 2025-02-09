###### Creating a Random Tensor

import torch

# Create a tensor with a 1*10 shape with a random value
# between 0 and 1.
r0 = torch.rand(10)

print(r0)
print("************************************************")


# Create a tensor with 10*1 shape with random value 
# between 0 and 1.
r1 = torch.rand((10, 1))
print(r1)
print("************************************************")


# Create a tensor with 2*2 shape with random value
# between 0 and 1.
r2 = torch.rand((2, 2))
print(r2)
print("************************************************")


# Create a tensor with 2*2 shape with random value
# from a normal distribution.
r3 = torch.randn((2, 2))
print(r3)
print("************************************************")


# Create an integer type tensor with 3*3 shape with random
# value between 0 and 10.
r4 = torch.randint(high=10, size=(3, 3))
print(r4)
print("************************************************")


# Create an integer type tensor with 3*3 shape with random
# value between 5 and 10.
r5 = torch.randint(low=5, high=10, size=(3, 3))
print(r5)
