###### Casting tensors into different device type

import torch



# Select the cpu device for running the tensor.
a = torch.tensor([1, 2, 3])
b = a.to('cpu')
print(b)


# Select the gpu device for running the tensor.
# Cast the device type if there is only one GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = torch.tensor([4, 5, 6])
c2 = c.to(device)
print(c2)
print(device)