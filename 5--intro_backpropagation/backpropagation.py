##
## 1) Forward Pass: Compute the Loss
## 2) Compute the local gradients
## 3) Backward Pass: Compute dLoss/dWeights using the chain rule
## 4) Update the weights
##

##
## Libraries
##
import torch

##
## GPU -- CUDA
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##
## Create a tensor to store the weights.
## Weights are the parameters that we want to learn.
##
weights = torch.ones(4, requires_grad=True, device=device)

##
## Create a tensor to store the inputs.
## Inputs are the data that we want to train on.
##
inputs = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)

##
## Create a tensor to store the outputs.
## Outputs are the data that we want to predict.
##
outputs = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device=device)

##
## The learning rate is a hyperparameter that controls how much
## we adjust the weights of our model with respect the loss gradient.
##
lr: float = 0.01

##
## Training loop -- Number of epochs (loops)
##
epochs: int = 1000


##
## Loss function -- This is the function that we want to minimize
## We want to minimize the loss function to make our model more accurate.
##
def loss_fn(weights, inputs, outputs):
    ##
    ## Absolute function -- Custom
    ##
    def square(x):
        return x * x

    ##
    ## Make a prediction. This works by using the adjusted
    ## weights against the inputs.
    ##
    prediction = weights * inputs

    ##
    ## Calculate the loss. This calculates the positive difference
    ## between the prediction and the actual value.
    ##
    ## Since we need a scalar value for the backward function
    ## to work, we will take the average (mean) of the differences.
    ##
    return square(prediction - outputs).mean()

    ##
    ## End of loss_fn
    ##


##
## Training loop
##
for i in range(epochs):
    ## Forward pass (calculate loss)
    model_output = loss_fn(weights, inputs, outputs)

    ## Backward pass (calculate gradients)
    model_output.backward()

    ## Using the output gradients, we can adjust the weights to minimize the loss
    with torch.no_grad():
        weights -= lr * weights.grad

    ##
    ## Zero the gradients after updating the weights
    ##  This is because the gradients are accumulated and we don't
    ##  want to double count them.
    ##
    weights.grad.zero_()

    ##
    ## End of training loop
    ##

##
## Print the reults
##
print(f"Predictions: {(weights * inputs).data.numpy()}")
print(f"Actual: {outputs.data.numpy()}")

##
## Output:
##  Predictions: [4.9999886 5.999997  6.999998  7.9999995]
##  Actual: [5 6 7 8]
##
## We can see that the predictions are very close to the actual values.
## This is because we adjusted the weights continuously to minimize the loss
## function. This incremently made the model more accurate.
##

##
## End of file
##
