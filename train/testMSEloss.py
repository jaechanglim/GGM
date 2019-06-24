import torch
import torch.nn as nn

a1 = torch.randn(4)
a2 = torch.randn(4)
criteria = nn.MSELoss()
loss = criteria(a1, a2)
print(loss)

diffs = []
for i in range(4):
    diff = a1[i] - a2[i]
    diffs.append(diff ** 2)
print(sum(diffs)/len(diffs))