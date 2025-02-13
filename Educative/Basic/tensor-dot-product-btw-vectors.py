###### Dot product between vectors


import torch

vectorT19 = torch.tensor([1, 2, 3, 4])
vectorT20 = torch.tensor([2, 3, 4, 5])

resultD = vectorT19.dot(vectorT20)
print("The result is \n {}".format(resultD))


print("="*30)
resultD = torch.dot(vectorT19, vectorT20)
print("The result is \n {}".format(resultD))
