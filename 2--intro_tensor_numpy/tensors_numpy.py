##
## This is the main file for the project
##
import torch
import numpy as np

# Create a tensor -- This is a type of matrix.
a = torch.ones(1, 5)  # rows, columns
print(a)

# We can convert a tensor to a numpy array
b = a.numpy()
print(b)

# We can convert a numpy array to a tensor
c = torch.from_numpy(b)
print(c)


# Let's create a new numpy array
d = np.ones(10)
# Convert the array to a tensor
e = torch.from_numpy(d)
print(e)
