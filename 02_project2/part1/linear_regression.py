import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
   
    X_T = np.transpose(X)
    I_matrix = np.identity(X.shape[1])
    matrix_inv = np.linalg.inv(np.dot(X_T, X) + lambda_factor * I_matrix)
    theta =  np.dot(matrix_inv, np.dot(X_T, Y))
    
    
    return theta
    # raise NotImplementedError

# X = np.array([[0.10252152, 0.3116275, 0.75139667, 0.96490904],
#  [0.98104447, 0.53182999, 0.74142857, 0.10707851],
#  [0.62126309, 0.37645494, 0.09231746, 0.87364862],
#  [0.24994759, 0.62797463, 0.99282863, 0.61622558],
#  [0.60352312, 0.26131664, 0.44118084, 0.97539724],
#  [0.98570034, 0.66665054, 0.19480128, 0.01619357],
#  [0.3188982,  0.62614334, 0.44107401, 0.6864829],
#  [0.28105923, 0.32048562, 0.23441847, 0.4663865],
#  [0.34763437, 0.15345648, 0.90538487, 0.49622285],
#  [0.82834703, 0.94605824, 0.38722846, 0.78779272],
#  [0.3794482,  0.27760235, 0.51867378, 0.2926096],
#  [0.46734934, 0.63153386, 0.34081298, 0.87864857],
#  [0.46059517, 0.49140508, 0.42220069, 0.61028108],
#  [0.92600171, 0.5799485,  0.70783939, 0.42639542],
#  [0.23770186, 0.33310088, 0.05336599, 0.1355399],
#  [0.33664733, 0.53857569, 0.50109788, 0.95083394],
#  [0.33770818, 0.56014034, 0.27237541, 0.89369849],
#  [0.23950367, 0.58848412, 0.15425427, 0.88151907]])

# Y = np.array([0.76696561, 0.19984546, 0.38571548, 0.39888696, 0.60476967, 0.69870741,
#  0.26846899, 0.41158194, 0.67407238, 0.91199785, 0.74846765, 0.62726223,
#  0.36298443, 0.7653095,  0.07394085, 0.92330298, 0.65669037, 0.07570072])

# lambda_factor = 0.6028165527852359

# result = closed_form(X, Y, lambda_factor)
# print(result)
### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
