import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from Dataset.mnist import load_mnist

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    train_size = x_train.shape[0]  # 훈련 데이터 - mnist의 경우 60,000개
    
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    # 0 ~ (train_size-1) 중 랜덤하게 batch_size개 선택
    
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    print(x_batch)
    print(t_batch)