# Understanding the Basics of a Neural Network

A neural network is a computational model inspired by the human brain. It consists of a large number of simple, highly interconnected processing nodes, or "neurons," which process information by their dynamic state response to external inputs.

A typical neural network consists of three types of layers:

- Input Layer:

This is where the network takes in feature values. Each neuron in this layer corresponds to one feature in the dataset.

- Hidden Layer(s):
  This layer, or these layers, perform computations on the weighted inputs and pass that information onto the next layer. Most of the complex computations of the neural network happen here.

- Output Layer:

The final layer produces the result for given inputs.

In this section, we will be building a simple feed-forward neural network from scratch, also known as a multi-layer perceptron (MLP), with one hidden layer. We'll use Python for this. The goal is to illustrate the core concepts that go into making a simple neural network.

## Neuron and Activation Function

Each neuron takes in a set of inputs, applies weights to them, adds a bias, and passes the result through an activation function. For our network, we will be using the sigmoid function as our activation function due to its nice properties such as its output being in the range of 0 to 1.

Here is a Python function that represents the sigmoid activation function:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

And its derivative, which we will use later in backpropagation:

```python
def sigmoid_derivative(x):
    return x * (1 - x)
```

Building a Simple Neural Network

We'll start by defining the structure of our neural network. For simplicity, our network will have two inputs, one hidden layer with two neurons, and an output layer with one neuron. All neurons will use the sigmoid activation function.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 2)
        self.weights2   = np.random.rand(2, 1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
```

Here's what's happening in the `__init__` method:

`self.input` is our input data.
`self.weights1` are the weights for the first layer of the network. We randomly initialize them.
`self.weights2` are the weights for the second layer of the network. We also randomly initialize them.
`self.y` is our target data.
`self.output` is the output data from the forward pass of our network. We initialize it to be the same shape as `self.y`.

In the next section, we'll discuss the forward propagation process and how the neural network uses the input data to give an output.

## Forward Propagation

Once we've set up the basic structure of our neural network, we need to define how data flows through it. This is known as forward propagation. During this step, the network makes its best guess about the output given the input.

```python

def feedforward(self):
    self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    self.output = sigmoid(np.dot(self.layer1, self.weights2))
```

In the feedforward method:

We start by passing the weighted sum of the inputs to the first layer of neurons and apply the sigmoid function.

We then pass the output of the first layer neurons (after applying the activation function) to the second layer neurons, again taking a weighted sum and applying the sigmoid function.

## Cost Function

Once the network has made a prediction, we need a way to measure how good that prediction is; i.e., how close is the network's output to the actual target value. This is typically done using a cost function (also known as a loss function). A common choice for a cost function in a neural network is the sum of squares error, which calculates the difference between the network output and the actual value for each output neuron, squares it, and sums all these squared errors together.

```python

def loss(self):
    return 0.5 * np.sum((self.y - self.output) ** 2)
```

In the loss method:

We calculate the half of squared difference between the target output (`self.y`) and predicted output (`self.output`), and sum all these values together to get our loss.

By adding these methods to our `NeuralNetwork` class, we now have a neural network that can make predictions and evaluate how well it's doing. However, the weights of the network are still random. In the next part of the tutorial, we'll cover how the neural network can learn from its mistakes to improve its predictions, a process known as backpropagation.

## Backpropagation

Backpropagation is the core algorithm for training neural networks. Once we have a measure of the error (via the cost function), we need to find a way to propagate this error back through the network to adjust the weights and bias. This is done using gradient descent, a process which finds the gradient (or slope) of the cost function for each weight with respect to a small change in that weight, then adjusts the weight in the direction that most decreases the cost.

First, we'll need to calculate the derivative of the loss function. The derivative of a function gives you the slope of the function at any point, which can be used to find local minima (the points where the function is at its lowest). Here's how we can implement this:

```python
def backprop(self):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
    d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

    # update the weights with the derivative (slope) of the loss function
    self.weights1 += d_weights1
    self.weights2 += d_weights2
```

In the backprop method:

We calculate the derivative of the loss function with respect to the weights (`d_weights2`, `d_weights1`). The full explanation of the mathematics here involves the chain rule of calculus and is a bit beyond the scope of this tutorial, but in essence, we're finding out how much a small change in the weights will change the output (and therefore the error).

We then update the weights in the direction that most reduces the error.

By adding the backprop method to our `NeuralNetwork` class, we now have a neural network that can not only make predictions (with `feedforward`) and evaluate its loss (with `loss`), but also learn from its mistakes to improve its predictions (with `backprop`). In the final part of this tutorial, we'll show you how to use these components to train the network.

## Training the Neural Network

Now that we've implemented both forward and backward propagation, we can combine them into a training process. During training, we repeatedly perform forward propagation, calculate the loss, perform backpropagation, and update the weights. Here is how we can do it:

```python
def train(self, X, y):
    self.input = X
    self.y = y
    for _ in range(1000): # iterations
        self.feedforward()
        self.backprop()
```

In the train method:

We first assign the input and target output.
We then use a loop to repeat the process a certain number of times (1000 iterations in this case).
In each iteration, we call self.feedforward() to make a prediction, and then self.backprop() to update the weights.
Finally, let's bring everything together and make a simple prediction:

```python
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
```

Here, X is a simple binary table and y is an XOR of X. The output will be the XOR of the input. With this simple neural network, you can see the output becoming closer to `[0, 1, 1, 0]` as it learns from the XOR function.

In this chapter we have built a simple neural network from scratch in Python, which can be trained using the backpropagation algorithm. This forms the foundation of many modern deep learning models. From here, you could extend this basic model into more complex types of neural networks. As an exercise, you might consider adding more layers, neurons, or different types of activation functions.
