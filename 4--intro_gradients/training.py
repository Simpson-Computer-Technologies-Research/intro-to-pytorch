"""
In summary, this code represents a basic training loop of a neural 
network where the model is making predictions (forward pass), calculating 
how wrong these predictions are (backward pass), and then updating its 
weights in an attempt to improve its predictions in the next epoch. The 
loop repeats this process for a specified number of epochs, with the goal 
of continuously reducing prediction error by fine-tuning the weights.
"""

##
## Libraries
##
import torch

##
## Create a tensor to store the weights.
## Weights are the parameters that we want to learn.
##
weights = torch.ones(4, requires_grad=True)

##
## The learning rate is a hyperparameter that controls how much
## we adjust the weights of our model with respect the loss gradient.
##
## A smaller learning rate makes the training process more gradual and
## precise, while a larger learning rate speeds up training but can
## overshoot the optimal weights.
##
learning_rate: float = 0.1

##
## Training loop -- Number of epochs (loops)
##
epochs: int = 3

##
## Training loop
##
for epoch in range(epochs):
    ##
    ## Forward pass (calculate output)
    ## This is where the model makes a prediction.
    ##
    ## In a more complex neural network, we would have more layers
    ## and more complex operations. model_output is the output of the
    ## neural network.
    ##
    model_output = (
        weights * 3
    ).sum()  # Sum aggregates these values into a single output (scalar)
    print(model_output)

    ##
    ## Backward pass (calculate gradients)
    ## This is where the model learns by adjusting the weights
    ##
    ## The gradient of a function at a point gives the direction with
    ## the steepest increase of the function at that point. The function
    ## is typically a loss function that we want to minimize.
    ##
    ## Here we pretend that our loss function is simply the model_output
    ## and we want to minimize it. When we multiply the weights by 3,
    ## the gradient is 3. (since the derivative of 3x is 3)
    ##
    ## If we performed (weights**2) * 3 + 1, the gradient would be 6. (power rule)
    ##
    ## Since model_output is not a loss but a direct function of the weights,
    ## model_output.backward() calculates and stores the gradients of this
    ## output with respect to weights.
    ##
    ## This is a problem because we want to minimize the loss, not the output.
    ## We can fix this by creating a loss function and then calling
    ## loss.backward() instead. This will calculate how we need to adjust our weights
    ## and biases to produce a lower loss.
    ##
    model_output.backward()  # Calculate gradients
    model_output_grad = weights.grad  # Pointer object

    ##
    ## Update the weights. This is the actual training step.
    ## We multiply the gradients by a learning rate (0.1)
    ##
    ## We want to temporarily disable gradient tracking,
    ## otherwise we would get an error.
    ##
    with torch.no_grad():
        weights -= learning_rate * model_output_grad

    ##
    ## Set the gradients to zero for the next iteration
    ## so that they don't accumulate over each epoch.
    ##
    ## Otherwise, the weights gradient will be 3, 6, 9, 12, ...
    ## But we want it to be 3, 3, 3, 3, ...
    ##
    model_output_grad.zero_()

##
## Print the final weights
##
print(weights)
