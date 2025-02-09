###### Creating a Tensor from a List

import torch

scalar = torch.tensor([10])
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1], [2], [3]])
matrices = torch.tensor([
    [[4],[5],[6]],
    [[7],[8],[9]],
    [[11],[13],[15]]
])
vector_of_matrices = torch.tensor([
    [[
        [16],[25],[36],
    ]],
    [[
        [49],[64],[81]
    ]],
    [[
        [121],[169],[225]
    ]],
    [[
        [20],[30],[42]
    ]],
    [[
        [56],[72],[90]
    ]],
    [[
        [132],[182],[240]
    ]],
])


print(scalar)

print(vector)

print(matrix)

print(matrices)

print(vector_of_matrices)


