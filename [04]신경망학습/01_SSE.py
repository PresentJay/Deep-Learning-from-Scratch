import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

if __name__ == "__main__":
    # 정답은 2 : one hot coding
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
    # 예시 1 : 정답을 2로 추정한 결과
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(sum_squares_error(y, t))
    
    # 예시 2 : 정답을 7로 추정한 결과
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(sum_squares_error(y, t))