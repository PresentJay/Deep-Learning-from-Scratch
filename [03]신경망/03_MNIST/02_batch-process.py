import sys, os, pickle
DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(DIR)
from Dataset.mnist import load_mnist
from functions import *
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(DIR + "/Dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    
    return softmax(a3)

if __name__ == "__main__":
    
    x, t = get_data()
    network = init_network()
    
    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x), batch_size):
        x_batch = x[i: i + batch_size]
        y_batch = predict(network, x_batch)
        
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
          
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))