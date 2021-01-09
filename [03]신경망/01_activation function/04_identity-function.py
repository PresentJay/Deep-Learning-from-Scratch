import numpy as np

def identity_function(x):
    return x

if __name__ == "__main__":
    # identity function returns input through
    X = np.array([1.0, 0.5])
    Y = identity_function(X)
    
    print(Y)