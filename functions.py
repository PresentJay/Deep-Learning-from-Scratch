import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
    
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t, onehot=True):
    # 데이터가 1차원일 경우
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    if onehot==True:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t]+ 1e-7)) / batch_size

def numerical_diff(f, x):
    h = 1e-4  # 0.0001 = (10 ^ (-4))
    return (f(x+h) - f(x-h)) / (2*h)