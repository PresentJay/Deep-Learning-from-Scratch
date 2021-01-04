import numpy as np

def AND(arr_x):
    weight = np.array([0.5, 0.5])
    bias = -0.7
    tmp = np.sum(weight*arr_x) + bias
    
    # print(tmp)
    # tmp is -0.1999 ... (approximately -0.2) : calculation error caused by floating point
    
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(arr_x):
    weight = np.array([-0.5, -0.5])
    bias = 0.7
    tmp = np.sum(weight*arr_x) + bias
    
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    x = np.array([0, 1])
    print(AND(x))
    print(NAND(x))
    