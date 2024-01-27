##
## Libraries
##
from torch import empty
from typing import Tuple


##
## Empty Matrix Class
##
class EmptyMatrix:
    ##
    ## Constructor
    ##
    def __init__(self, dtype, size: Tuple):
        self.dtype = dtype
        self.size = size
        self.matrix = empty(size, dtype=dtype)

        ##
        ## End of Constructor
        ##

    ##
    ## Get the matrix
    ##
    def get(self):
        return self.matrix

        ##
        ## End of get function
        ##

    ##
    ## Get the matrix size
    ##
    def __len__(self):
        return len(self.size)

        ##
        ## End of len function
        ##

    ##
    ## Add another matrix to this matrix
    ##
    def __add__(self, other):
        self.matrix.add(other.matrix)
        return self

        ##
        ## End of add function
        ##

    ##
    ## End of Empty Matrix class
    ##


##
## End of file
##
