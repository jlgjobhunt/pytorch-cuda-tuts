# Casting tensors into different types
import torch


# torch.bool
a = torch.tensor([True], dtype=torch.bool)
print("The dtype for a is {}".format(a.dtype))

# torch.bool to torch.int8
a2 = a.to(dtype=torch.int8)
print("The dtype for a2 is {}".format(a2.dtype))

# torch.int8
b = torch.tensor([8], dtype=torch.int8)
print("The dtype for b is {}".format(b.dtype))

# torch.int8 to torch.int16
b2 = b.to(dtype=torch.int16)
print("The dtype for b2 is {}".format(b2.dtype))

# torch.int16
c = torch.tensor([16], dtype=torch.int16)
print("The dtype for c is {}".format(c.dtype))

# torch.int16 to torch.int32
c2 = c.to(dtype=torch.int32)
print("The dtype for c2 is {}".format(c2.dtype))

# torch.int32
d = torch.tensor([32], dtype=torch.int32)
print("The dtype for d is {}".format(d.dtype))

# torch.int32 to torch.int64
d2 = d.to(dtype=torch.int64)
print("The dtype for d2 is {}".format(d2.dtype))

# torch.int64
e = torch.tensor([64], dtype=torch.int64)
print("The dtype for e is {}".format(e.dtype))

# torch.int64 to torch.uint64
e2 = e.to(dtype=torch.uint64)
print("The dtype for e2 is {}".format(e2.dtype))

# torch.uint8
f = torch.tensor([8], dtype=torch.uint8)
print("The dtype for f is {}".format(f.dtype))

# torch.uint8 to torch.int8
f2 = f.to(dtype=torch.int8)
print("The dtype for f2 is {}".format(f2.dtype))

# torch.uint16
g = torch.tensor([16], dtype=torch.uint16)
print("The dtype for g is {}".format(g.dtype))

# torch.uint16 to torch.int16
g2 = g.to(dtype=torch.int16)
print("The dtype for g2 is {}".format(g2.dtype))

# torch.uint32
h = torch.tensor([32], dtype=torch.uint32)
print("The dtype for h is {}".format(h.dtype))

# torch.uint32 to torch.int32
h2 = h.to(dtype=torch.int32)
print("The dtype for h2 is {}".format(h2.dtype))

# torch.uint64
i = torch.tensor([64], dtype=torch.uint64)
print("The dtype for i is {}".format(i.dtype))

# torch.uint64 to torch.int64
i2 = i.to(dtype=torch.int64)
print("The dtype of i2 is {}".format(i2.dtype))

# torch.float16
j = torch.tensor([16], dtype=torch.float16)
print("The dtype for j is {}".format(j.dtype))

# torch.float16 to torch.uint8
j2 = j.to(dtype=torch.uint8)
print("The dtype for j2 is {}".format(j2.dtype))

# torch.float32
k = torch.tensor([32], dtype=torch.float32)

# torch.float32 to torch.uint8
k2 = k.to(dtype=torch.uint8)
print("The dtype for k2 is {}".format(k2.dtype))

# torch.float64
l = torch.tensor([64], dtype=torch.float64)
print("The dtype for l is {}".format(l.dtype))

# torch.float64 to torch.uint8
l2 = l.to(dtype=torch.uint8)
print("The dtype for l2 is {}".format(l2.dtype))

