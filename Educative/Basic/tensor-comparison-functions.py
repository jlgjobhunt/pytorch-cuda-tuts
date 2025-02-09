# Comparison Functions


import torch


# Original Tensor
vectorT15 = torch.randn((3,4))
print("The original tensor is \n {}".format(vectorT15))

# Print Line of Demarcation
print("="*30)

print("The comparison between a tensor and a single value.\n")


# Print Line of Demarcation
print("="*30)

# Comparison Tensor - Less Than (torch.lt)
comparisonT16 = torch.lt(vectorT15, 0.5)

print("The element is less than 0.5 \n {}".format(comparisonT16))

# Print Line of Demarcation
print("="*30)


# Tensor Vector of random floats between 3 and 4.
vectorT17 = torch.randn((3, 4))

print("The comparison between two tensors.\n")

# Print Line of Demarcation
print("="*30)


# Comparison Tensor - Greater Than (torch.gt)
comparisonT18 = torch.gt(vectorT15, vectorT17)
print("The comparison result between tensor vectorT15 & vectorT17: \n {}".format(comparisonT18))


# Print Line of Demarcation
print("="*30)