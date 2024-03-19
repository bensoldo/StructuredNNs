import numpy as np
import torch
from strnn.models.strNN import StrNN

# print(A)
A = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
])

print(A)

out_dim = A.shape[0]
in_dim = A.shape[1]
hid_dim = (50, 50)

strnn = StrNN(in_dim, hid_dim, out_dim, opt_type='greedy', adjacency=A)

print(strnn)

x = torch.randn(in_dim)
print(x)
y = strnn(x)