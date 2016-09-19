import sys
sys.path.append('../source')

import network
import numpy as np

# data
train = [
    (np.array([[0],[0]]), np.array([[1],[1]])),
    (np.array([[0],[1]]), np.array([[1],[0]])),
    (np.array([[1],[0]]), np.array([[0],[1]])),
    (np.array([[1],[1]]), np.array([[0],[0]]))
]
test = [
    (np.array([[0],[0]]), np.array([[1],[1]])),
    (np.array([[0],[1]]), np.array([[1],[0]])),
    (np.array([[1],[0]]), np.array([[0],[1]])),
    (np.array([[1],[1]]), np.array([[0],[0]])),
]

data = [
    np.array([[0],[0]]),
    np.array([[0],[1]]),
    np.array([[1],[0]]),
    np.array([[1],[1]])
]

# set up
net = network.net([2, 2, 2])

# train
net.train(train, 50, 2, 5, test)

# predict
result = net.predict( data )