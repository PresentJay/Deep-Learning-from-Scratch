import numpy as np

if __name__ == "__main__":
    A = np.array([[1,2], [3,4]])
    B = np.array([[5,6], [7,8]])
    
    C = np.array([[1,2,3], [4,5,6]])
    D = np.array([[1,2], [3,4], [5,6]])
    
    # dot function calculates matrix multiplication
    print(np.dot(A,B))
    print(np.dot(C,D))
    
    # you must concern about each array's shapes.
    # ex01 : (2,3) X (3,4) = (2,4)   : 2,   (3 = 3,)   4
    # ex02 : (1,5) X (5,5) = (5,5)   : 1,   (5 = 5,)   5
    # ex03 : (2,5) X (3,3) = error   : 2,   (5 != 3,)  3
    # ex04 : (2,3) X (2,2) = error   : 2,   (3 != 2,)  2
    # ex05 : (2,3) X (3,2) = (2,2)   : 2,   (3 = 3,)   2