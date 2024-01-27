##
## Libraries
##
from torch import rand
from typing import Tuple


##
## Random Matrix class
##
class RandomMatrix:
    ##
    ## Constructor
    ##
    def __init__(self, dtype, size: Tuple):
        self.dtype = dtype
        self.size = size
        self.matrix = rand(size, dtype=dtype)

        ##
        ## End of Constructor
        ##

    ##
    ## Get the matrix
    ##
    def get(self):
        return self.matrix

    ##
    ## Get the matrix size
    ##
    def __len__(self):
        return len(self.size)

    ##
    ## Add another matrix to this matrix
    ##
    def __add__(self, other):
        self.matrix.add(other.matrix)
        return self

    ##
    ## End of Random Matrix class
    ##


##
## End of file
##
