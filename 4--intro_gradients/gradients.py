import torch

##
## The gradient is the slope of a function at a given point.
## It is used to determine how much the parameters of a function
## need to change to minimize the loss of the function.
##
## It's basically the derivative of a function at a given point.
##
x = torch.randn(3, requires_grad=True)
print(x)

##
## We can do operations on the tensor
##
y = x + 2
print(y)

##
## We can also do more operations on the tensor
##
## This will take the tensor and multiply it by itself and then multiply
## it by 2. This will return a tensor with the same shape as the original
## tensor. (A vector with 3 values)
z = y * y * 2

## For the background function to work, we must have a scalar value!
z = z.mean()  # This will take the average of all of the values in the tensor
print(z)

##
## Backpropagate -- This is the process of calculating the gradient
## of a function at a given point.
##
## Backpropogation calculates how much the parameters need to change
## to minimize the loss of the function.
z.backward()  # dy/dx

##
## Print the gradient. This is a vector of the gradient of the function
## at a given point.
##
print(x.grad)
