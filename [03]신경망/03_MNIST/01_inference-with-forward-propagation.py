import sys, os
from mnist import load_mnist
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from functions import *
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == "__main__":
    (x_train, t_train), (x_text, t_text) = load_mnist(flatten=True, normalize=False)
    # flatten parameter makes input images simple array
    # normalize parameter normalizes input image's pixels to between 0.0 and 1.0 (or it can be 0~255)
    
    print(x_train.shape)
    print(t_train.shape)
    print(x_text.shape)
    print(t_text.shape)
    
    img = x_train[0]
    label = t_train[0]
    print(label)
    
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    
    img_show(img)