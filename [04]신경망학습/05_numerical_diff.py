import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4  # 0.0001 = (10 ^ (-4))
    return (f(x+h) - f(x-h)) / (2*h)
    # 중앙차분, 중심차분

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    
    # x의 특정 값에서의 기울기를 볼 수 있는 변수
    a = 10
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    tf = tangent_line(function_1, a)
    y2 = tf(x)
    
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.show()