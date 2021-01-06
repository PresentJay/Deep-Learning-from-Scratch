import numpy as np

if __name__ == "__main__":
    A = np.array([1,2,3,4])
    B = np.array([[1,2], [3,4], [5,6]])
    
    print(A)
    print(B)
    
    # ndim( <array> ) returns a number of dimention
    print(np.ndim(A))
    print(np.ndim(B))

    print(A.shape)
    print(B.shape)    
    
    print(A.shape[0])