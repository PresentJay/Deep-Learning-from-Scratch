import sys, os
DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(DIR)
from functions import *

# 편미분 : 변수가 여럿인 함수의 미분 : 편미분용 함수를 더 만들어서 풀면 된다!

def function(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)


# 문제A : x0=3, x1=4일 때 x0에 대한 편미분을 구하여라
# x0을 목표로, x1을 4.0으로 고정함
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

# 문제B : x0=3, x1=4일 때 x1에 대한 편미분을 구하여라
# x1을 목표로, x0을 3.0으로 고정함
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

if __name__ == "__main__":
    # 정답A
    print(numerical_diff(function_tmp1, 3.0))
    
    # 정답B
    print(numerical_diff(function_tmp2, 4.0))