# Specifying tensor types when creating tensors

import torch

a = torch.tensor([1, 2, 3])
print("The dtype for a is {}".format(a.dtype))

b = torch.tensor([1, 2, 3], dtype=torch.float)
print("The dtype for b is {}".format(b.dtype))

c = torch.tensor([False], dtype=torch.bool)
print("The dtype for c is {}".format(c.dtype))

d = torch.tensor([True], dtype=torch.bool)
print("The dtype for d is {}".format(d.dtype))

e = torch.tensor([8], dtype=torch.int8)
print("The dtype for e is {}".format(e.dtype))

f = torch.tensor([16], dtype=torch.int16)
print("The dtype for f is {}".format(f.dtype))

g = torch.tensor([32], dtype=torch.int32)
print("The dtype for g is {}".format(g.dtype))

h = torch.tensor([64], dtype=torch.int64)
print("The dtype for h is {}".format(h.dtype))

i = torch.tensor([8],dtype=torch.uint8)
print("The dtype for i is {}".format(i.dtype))

j = torch.tensor([16], dtype=torch.uint16)
print("The dtype for j is {}".format(j.dtype))

k = torch.tensor([32], dtype=torch.uint32)
print("The dtype for k is {}".format(k.dtype))

l = torch.tensor([64], dtype=torch.uint64)
print("The dtype for l is {}".format(l.dtype))

m = torch.tensor([16], dtype=torch.float16)
print("The dtype for m is {}".format(m.dtype))

n = torch.tensor([32], dtype=torch.float32)
print("The dtype for n is {}".format(n.dtype))

o = torch.tensor([64], dtype=torch.float64)
print("The dtype for o is {}".format(o.dtype))

