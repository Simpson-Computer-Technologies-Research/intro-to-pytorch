##
## This is the main file for the project
##
import torch
from random_matrix import RandomMatrix
from empty_matrix import EmptyMatrix
import numpy as np

##
## Create an empty matrix
##
# empty_matrix = EmptyMatrix(torch.float, (2, 2))
# empty_matrix += empty_matrix
# print(empty_matrix)

##
## Create a random matrix
##
random_matrix = RandomMatrix(torch.float, (2, 2))
random_matrix += random_matrix
print(random_matrix.get())

##
## View the matrix as a single row
##
## This will take all of the values in each row and put them into a single row
## We will then print the values in that row
##
print(
    random_matrix.get().view(1, 4)  # rows, columns
)  # The number must be the size of the matrix (4 = 2 * 2)

##
## View the matrix as a single column
##
## This will take all of the values in each column and put them into a single column
## We will then print the values in that column
##
print(
    random_matrix.get().view(4, 1)  # rows, columns
)  # The number must be the size of the matrix (4 = 2 * 2)
