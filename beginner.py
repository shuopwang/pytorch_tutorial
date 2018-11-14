import numpy as np
import torch
x=torch.Tensor(5,3)
print(x)
#create a matrix without init
x=torch.rand(5,3)
print(x)
#init randomly

print(x.size())
#return the shape with tuple

y=torch.rand(5,3)
print(x+y)

print(torch.add(x,y))

result=torch.Tensor(5,3)

torch.add(x,y,out=result)
print(result)

x=torch.randn(4,4)
y=x.view(16)
z=x.view(-1,8)
print(x.size(),y.size(),z.size())

print('-'*5+'from tensor to numpy'+'-'*5)
a=torch.ones(5)
b=a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

print('-'*5+'from numpy to tensor'+'-'*5)
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

print(torch.cuda.is_available())