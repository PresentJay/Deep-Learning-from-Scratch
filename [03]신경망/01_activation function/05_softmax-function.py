import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def advanced_softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

if __name__ == "__main__":
    a = np.array([0.3, 2.9, 4.0])
    print(softmax(a))
    
    b = np.array([1010, 1000, 900])
    print(softmax(b))
    # nan (overflow error errupted)
    
    print(advanced_softmax(b))