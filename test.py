import torch

a = torch.ones((1,3,3))
tmp = a[0, 1, :]
print(tmp)
t = torch.sigmoid(tmp)
for i in range(a.shape[0]):
    tmp[i] = t[i]
print(a)
