# Simple Neural Network
Python library built for learning and demonstration purposes.

Details
--
Feedforward neural network with stochastic gradient descent learning algorithm.  
Gradients are calculated using backpropagation.  

Naive implementation. 

Example
--
~~~~
import network

# data
train = [
    ([0,0], [1,1]),
    ([0,1], [1,0]),
    ([1,0], [0,1]),
    ([1,1], [0,0])
]
test = [
    ([0,0], [1,1]),
    ([0,1], [1,0])
]
data = [
    [1,0],
    [1,1]
]

# set up
net = network.net([2, 2, 2])

# train
net.train(train, 30, 2, 5.0, test)

# predict
result = net.predict( data )
~~~~

Version 0.1:
- basic neural network

