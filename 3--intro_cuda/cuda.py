import torch
import numpy as np

##
## Let's check if the computer has a gpu and if it does, we will use it.
## We can do this by checking if 'cuda' is available.
##
if torch.cuda.is_available():
    ##
    ## Define a device variable -- This will be used to tell the computer
    ## to use the gpu for specific tasks.
    ##
    device = torch.device("cuda")
    print("There is a GPU")

    ##
    ## Create a tensor -- This is a type of matrix.
    ##
    a = torch.ones(1, 5, device=device)  # rows, columns

    ##
    ## So why do we use cuda?
    ## We use cuda to speed up the process of training a neural network
    ##
    ## We can convert a tensor to a numpy array
    ## But we must first convert the tensor to the cpu since numpy doesn't
    ## support cuda and the gpu!
    ##
    b = a.to("cpu").numpy()  # We must first convert the tensor to the cpu
    # b = a.cpu().numpy()  # An alternative method
    print(b)

    ##
    ## If we want to optimize the speed of our neural network, we can also
    ## make it require a gradient.
    ##
    ## Gradients -- The slope of a function at a given point -- are used to
    ## optimize neural networks and models.
    ##
    x = torch.ones(5, 5, device=device, requires_grad=True)
