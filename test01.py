import torch

a = torch.tensor([[1,2,3,34234243,4,2,2,1,6100]])

print(torch.argmax(a, dim=1))