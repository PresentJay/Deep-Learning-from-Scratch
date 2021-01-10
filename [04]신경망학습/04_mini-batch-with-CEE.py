import numpy as np

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

if __name__ == "__main__":
    pass