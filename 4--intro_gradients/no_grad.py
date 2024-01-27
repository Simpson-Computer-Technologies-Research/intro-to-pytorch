import torch

x = torch.randn(3, requires_grad=True)
print(x)

##
## We can disable the gradient 3 different ways
##
## 1. x.requires_grad_(False) -- This will disable the gradient
## 2. x.detach() -- This will detach the gradient
## 3. with torch.no_grad(): -- This will disable the gradient
##

## 1.
x.requires_grad_(False)  # This will disable the gradient

## 2.
x.detach()  # This will detach the gradient

## 3.
with torch.no_grad():  # This will disable the gradient
    y = x + 2
    print(y)
