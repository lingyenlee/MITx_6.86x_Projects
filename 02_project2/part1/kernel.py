import numpy as np

### Functions for you to fill in ###

def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    # kernal_matrix = (np.dot(X, np.transpose(Y)) + c) ** p 
    # return kernal_matrix
    kernel_matrix = np.zeros(shape=(X.shape[0], Y.shape[0]))
    

    for i, x in enumerate(X):
        for j, y in enumerate(Y):        
            kernel_matrix[i,j] = (np.dot(x,y) + c) ** p 
    return kernel_matrix
    raise NotImplementedError

# X = np.array([[9.44808862e-01, 8.47555677e-01],
#  [6.75473869e-01, 4.16229640e-01],
#  [7.37593554e-02, 5.08711259e-01],
#  [4.73287824e-01, 8.88567308e-01],
#  [7.07347211e-01, 2.98499796e-02],
#  [8.56658911e-01, 7.34149401e-01],
#  [4.40304912e-01, 6.11063197e-01],
#  [1.75651708e-01, 3.96600978e-02],
#  [3.59800519e-01, 2.67040814e-01],
#  [2.92683898e-06, 7.71928123e-01]])
# Y = np.array([[0.55008075, 0.1880837 ],
#  [0.88923195, 0.83214472],
#  [0.75068489, 0.5588624 ],
#  [0.32334572, 0.52078792],
#  [0.56551913, 0.79898971],
#  [0.12480328, 0.83240248],
#  [0.46270692, 0.58656552],
#  [0.97255563, 0.60253041],
#  [0.44756597, 0.12784042],
#  [0.83066467, 0.60719177]])
# c = 3
# p = 3

# result =  polynomial_kernel(X, Y, c, p)
# print(result)

def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    kernel_matrix = np.zeros(shape=(X.shape[0], Y.shape[0]))
    
    for i,x in enumerate(X):
       
        for j,y in enumerate(Y):
            
            kernel_matrix[i,j] = np.exp(-gamma * np.linalg.norm(x-y) **2)
    return kernel_matrix
    raise NotImplementedError
