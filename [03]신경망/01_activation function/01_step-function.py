import numpy as np
import matplotlib.pylab as plt

def step_function(x):
# if you implement like below code, you can't use numpy object.
    """ 
        if x>0:
            return 1
        else:
            return 0
    """
# then, you can make it like this below code.
    y = x > 0
    # numpy array's inequality operation make results about each member.
    return y.astype(np.int)
    # astype() returns numpy array's type
    # astype(<TYPE>) makes numpy array's type to <TYPE>
    # then, above code calculates x(numpy array)'s step function to boolean(false or true)
    # after that, make boolean to integer, so that results can just be 1 or 0.

if __name__ == "__main__":
    
    x = np.arange(-5.0, 5.0, 0.1)
    # arange function generates a numpy array that has members
    # at the 3rd argument interval
    # from the 1st argument to the 2nd argument
    # so it generates [-0.5, -0.4, -0.3, ... , 4.8, 4.9, 5.0]
    
    y = step_function(x)
    
    plt.plot(x, y)
    # set graph's parameter
    
    plt.ylim(-0.1, 1.1)
    # set limit of y axis
    
    plt.show()
    # show graph(plot)