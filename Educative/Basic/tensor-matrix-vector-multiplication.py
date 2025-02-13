# Matrix Vector Multiplication

import torch


# Matrix multiplication with vectors.


matrixTf = torch.ones((2, 4))

print("The matrix is: \n {}".format(matrixTf))

# Print Line of Demarcation
print("="*30)

vectorT18 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
print("The vector is: \n {}".format(vectorT18))

# Print Line of Demarcation
print("="*30)

resultB = matrixTf.mv(vectorT18)
print("The result is: \n {}".format(resultB))

