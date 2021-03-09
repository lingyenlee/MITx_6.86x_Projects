import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred
    raise NotImplementedError

# train_x= np.array([[0.80373532, 0.2219286 ],
#   [0.73841526, 0.07802308],
#   [0.70625986, 0.05515655],
#   [0.88901774, 0.70960285],
#   [0.10940147, 0.43631066],
#   [0.85773366, 0.70584257],
#   [0.55113232, 0.25925492],
#   [0.10274124, 0.13003314],
#   [0.21580622, 0.45219484],
#   [0.72996019, 0.05615394],
#   [0.87756327, 0.44398177],
#   [0.99896254, 0.77016728],
#   [0.58129048, 0.26210376],
#   [0.72311394, 0.74496437],
#   [0.19724657, 0.86225842],
#   [0.43765616, 0.46466816],
#   [0.45605635, 0.40188707]])
# train_y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1])
# test_x= np.array([[0.09574692, 0.84712555],
#   [0.07777195, 0.98835475],
#   [0.88264334, 0.73971992],
#   [0.02561285, 0.72167952],
#   [0.91508213, 0.13593894],
#   [0.54575486, 0.29739794],
#   [0.18174771, 0.0449108 ],
#   [0.74385003, 0.4025301 ],
#   [0.15533319, 0.18234483],
#   [0.24961207, 0.67263234],
#   [0.7325327,  0.17533492]])
# Submission output = [0 0 1 0 1 0 0 1 0 0 0]

# result = one_vs_rest_svm(train_x, train_y, test_x)
# print(result)

def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1, multi_class="ovr")
    clf.fit(train_x, train_y)
    pred_test_y = clf.predict(test_x)
    return pred_test_y
    raise NotImplementedError

# train_x= [[0.68169967, 0.0426867,  0.23342619],
#  [0.83833586, 0.57927197, 0.03114045],
#  [0.90416209, 0.5163992,  0.8021711 ],
#  [0.98104037, 0.36558555, 0.94525234],
#  [0.46299652, 0.21232264, 0.98720908],
#  [0.62696445, 0.48628602, 0.87304434],
#  [0.65524844, 0.04850744, 0.57665758],
#  [0.1587856,  0.92427845, 0.4666918 ],
#  [0.46873454, 0.27606745, 0.37493461],
#  [0.51430927, 0.34816576, 0.37637887],
#  [0.2194154,  0.16588386, 0.28742985]]
# train_y= [6, 6, 5, 4, 0, 4, 7, 1, 0, 6, 3]
# test_x= [[0.51496444, 0.06739121, 0.57277467],
#  [0.52107339, 0.75730827, 0.9085599 ],
#  [0.75729358, 0.87130896, 0.09209847],
#  [0.6556377,  0.37393544,0.55538302],
#  [0.37972805, 0.43552525, 0.51154571],
#  [0.04441269, 0.79922072, 0.00551599],
#  [0.80489321, 0.29226258, 0.35329993],
#  [0.57273369, 0.82588618, 0.64621544],
#  [0.88176536, 0.12379852, 0.19413536],
#  [0.05233315, 0.94771724, 0.27577675]]
# # Submission output: [6 4 6 6 6 6 6 6 6 6]

# result = multi_class_svm(train_x, train_y, test_x)
# print(result)



def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

