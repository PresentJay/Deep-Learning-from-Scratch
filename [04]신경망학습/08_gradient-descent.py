import numpy as np

def function(x):
    # return x[0]**2 + x[1]**2
    return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad

# f : 최적화하려는 함수
# init_x : 초기값
# lr : learning rate : 학습률
# step_num : 반복횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x

if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])
    
    # 실제 최소값은 0,0이고, 0,0에 가까운 결과를 냄!
    print(gradient_descent(function, init_x, 0.1, 100))
    
    # 학습률이 너무 작거나 커도 좋은 결과를 낼 수 없음
    
    # 학습률이 너무 작은 예
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function, init_x, lr=1e-10, step_num=100))
    
    # 학습률이 너무 큰 예
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function, init_x, lr=10.0, step_num=100))