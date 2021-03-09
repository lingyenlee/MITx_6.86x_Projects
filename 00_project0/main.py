import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    x = np.random.random_sample([n,1])
    return(x) 
    raise NotImplementedError


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.random([h, w])
    B = np.random.random([h, w])
    s = A + B
    return A, B, s
    
    raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    C = A + B
    s = np.linalg.norm(C)
    return s
    raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    # weights_transpose = weights.transpose()
    # out = weights_transpose * inputs
    inputs_arr = np.array(inputs)
    weights_arr = np.array(weights)
    weights_transpose = np.transpose(weights_arr)
    output = np.matmul(weights_transpose, inputs_arr)
    print(output.shape)
    out = np.tanh(output)
    return out
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x <= y:
        return x * y
    else:
        return x / y
    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    vfunc = np.vectorize(scalar_function)
    return vfunc(x,y)
    #Your code here
    raise NotImplementedError

