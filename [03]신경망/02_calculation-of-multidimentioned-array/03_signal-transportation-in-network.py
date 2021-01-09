import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from functions import *
import numpy as np

def init_network():
    network = {
        'W1' : np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        'b1' : np.array([0.1, 0.2, 0.3]),
        'W2' : np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        'b2' : np.array([0.1, 0.2]),
        'W3' : np.array([[0.1, 0.3], [0.2, 0.4]]),
        'b3' : np.array([0.1, 0.2]),
    }
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    
    return identity_function(a3)

if __name__ == "__main__":
    # print(test())
    
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
    
    
def test():
    # from input layer to first layer
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    
    # print(A1)
    # print(Z1)
    
    # from first layer to second layer
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    
    # print(A1)
    # print(Z1)
    
    # from second layer to output layer
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    
    A3 = np.dot(Z2, W3) + B3
    
    return identity_function(A3)