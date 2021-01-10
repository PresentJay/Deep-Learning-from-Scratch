import math
import numpy as np
import matplotlib.pyplot as plt

def log_graph_test(x):
    y = np.log(x)
    plt.plot(x, y)
    plt.ylim(-5, 0)
    plt.xlim(0.0, 1.0)
    plt.show()
    
def cross_entropy_error(y, t):
    # 0을 입력하면 log 0이 -inf(마이너스 무한대)이기 때문에, 아주 작은 값을 더해주는 작업
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))
    
if __name__ == "__main__":
    # x = np.arange(0.005, 1.1, 0.2)
    # log_graph_test(x)
    # 정답은 2 : one hot coding
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
    # 예시 1 : 정답을 2로 추정한 결과
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy_error(y, t))
    
    # 예시 2 : 정답을 7로 추정한 결과
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(cross_entropy_error(y, t))