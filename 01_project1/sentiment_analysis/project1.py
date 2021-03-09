from string import punctuation, digits
import numpy as np
import random
import utils
# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    
    L = np.max([0, 1 - label * (np.dot(theta, feature_vector) + theta_0)])
    return L
    raise NotImplementedError
    
    """ Provided answer """
    
    # Answer 1
    # y = np.dot(theta, feature_vector) + theta_0
    # loss = max(0.0, 1 - y * label)
    # return loss
    
    # Answer 2
    # y = theta @ feature_vector + theta_0
    # return max(0, 1 - y * label)
    
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    total_hinge = 0
    for k, v in enumerate(feature_matrix):
        
        # print(len(feature_matrix))
        total_hinge += np.max([0, 1 - labels[k] * (np.dot(theta, v) + theta_0)])/len(feature_matrix)
    return total_hinge
    
    """provided answer  """
    # ys = feature_matrix @ theta + theta_0
    # loss = np.maximum(1 - ys * labels, np.zeros(len(labels)))
    # return np.mean(loss)
    
    
    raise NotImplementedError
#pragma: coderesponse end

# feature_matrix = np.array([[0.75581008, 0.60751138, 0.55978624, 0.81728483, 0.9363876, 0.31136247, 0.73302413, 0.09813534, 0.01327215, 0.17528011], 
#                             [0.02872649, 0.38819003, 0.70087008, 0.60890708, 0.12397982, 0.33028173, 0.94064299, 0.78820814, 0.22887465, 0.9997299], 
#                             [0.38594582, 0.44330319, 0.93235809, 0.43764238, 0.60184255, 0.9996483,  0.56348055, 0.27839384, 0.82801782, 0.05930217], 
#                             [0.60242837, 0.97669626, 0.12752898, 0.23393266, 0.19866332, 0.16314239, 0.5289254,  0.7551023,  0.02386321, 0.77059032], 
#                             [0.25033218, 0.97028919, 0.24179327, 0.35239397, 0.73771929, 0.00238174, 0.79280619, 0.1004359,  0.78806697, 0.87996078], 
#                             [0.36173412, 0.37787042, 0.88955761, 0.26206947, 0.54927984, 0.13705737, 0.56011235, 0.9623955,  0.21888103, 0.81549023], 
#                             [0.73158767, 0.74873427, 0.6417947,  0.04100103, 0.62763214, 0.05959359, 0.86526654, 0.02592014, 0.91854128, 0.10084347], 
#                             [0.82551338, 0.55350013, 0.21745442, 0.39210811, 0.97142927, 0.29752198, 0.95698601, 0.28501298, 0.43334376, 0.22049526], 
#                             [0.94793891, 0.16950306, 0.96819852, 0.4788734,  0.62127669, 0.92899082, 0.71339565, 0.25254073, 0.01097113, 0.73004027], 
#                             [0.09095118, 0.80187446, 0.22692378, 0.99746702, 0.69163848, 0.28747962, 0.00620788, 0.98789068, 0.52393372, 0.18701445]])
# labels = np.array([0, 0, 2, 1, 0, 1, 1, 1, 2, 0])
# theta = np.array([0.38822878, 0.61706506, 0.94039517, 0.11257146, 0.04532565, 0.98305659, 0.56897019, 0.13955828, 0.02530833, 0.56250655])
# theta_0 = 0

# result = hinge_loss_full(feature_matrix, labels, theta, theta_0)
# print(result)


#pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    if (np.dot(current_theta, feature_vector) + current_theta_0) * label <= 0:
        current_theta = current_theta + feature_vector * label
        current_theta_0 = current_theta_0 + label

    return (current_theta, current_theta_0)

    """ provided answer """
    # if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1e-7:
    #     return (current_theta + label * feature_vector, current_theta_0 + label)
    # return (current_theta, current_theta_0) 
    
raise NotImplementedError
#pragma: coderesponse end

# feature_vector= np.array([-0.14212493, 0.22611762, -0.17151451, -0.1218966, 0.44142949, -0.10564561, -0.32276396, 0.36108784, -0.20122086, -0.19091659])
# label= -1
# current_theta= np.array([ 0.03275864, 0.23264252, 0.13864774, 0.45379014, -0.16857629, 0.14648217, -0.38125299, -0.38036766, 0.25921491, 0.11719199])
# current_theta_0= 0.12245768645522437
# perceptron_single_step_update output is (['0.0327586', '0.2326425', '0.1386477', '0.4537901', '-0.1685763', '0.1464822', '-0.3812530', '-0.3803677', '0.2592149', '0.1171920'], '0.1224577')

# result = perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0)
# print(result)

#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    theta = np.zeros(shape=feature_matrix.shape[1])
    theta_0 = 0
  
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            
            result = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            
            # use the updated theta/theta_0 for next iter
            theta = result[0]
            theta_0 = result[1]

    return (theta, theta_0)

    """ provided answer """
    # (nsamples, nfeatures) = feature_matrix.shape
    # theta = np.zeros(nfeatures)
    # theta_0 = 0.0
    # for t in range(T):
    #     for i in get_order(nsamples):
    #         theta, theta_0 = perceptron_single_step_update(
    #             feature_matrix[i], labels[i], theta, theta_0)
    # return (theta, theta_0)
    raise NotImplementedError
#pragma: coderesponse end

# feature_matrix= np.array([
#   [ 0.43351817,  0.09695823, -0.10279196, -0.02533037, 0.48024837, 0.24009827, 0.06409088, 0.0749311, 0.23016897, -0.19218163],
#   [ 0.15425097,  0.35548244, -0.43279263, -0.20129779, -0.3551962,  -0.41100928, 0.21146015, -0.03496779, -0.37929887, -0.06860828],
#   [ 0.40182153, -0.18211767, 0.44230463, -0.18627085,  0.22812381,  0.08813151, 0.08988961,  0.15557452, -0.19040053,  0.24792025],
#   [ 0.24799512,  0.12365608, 0.01198942,  0.28107166, -0.47264926, -0.05735119, 0.31578395, -0.26545544, -0.42702048,  0.22683414],
#   [ 0.14042205, -0.49037975, 0.43048589, -0.03270358, -0.43414672, -0.30322815, -0.27262686, -0.3981654,  -0.17925828,  0.1152774 ]])
# labels= [-1,  1, -1, -1,  1]
# T = 5
# # perceptron output is ['-0.3551436', '-0.0764357', '-0.4566008', '-0.3288022', '-0.5448175', '-0.7450177', '-0.4668403', '-0.3232523', '0.0588639', '-0.4280853']

# result = perceptron(feature_matrix, labels, T)
# print(result)

#pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    # number of samples = rows = feature_matrix.shape[0]
    # number of features = columns = feature_matrix.shape[1]
    
    n = feature_matrix.shape[0]
    
    # initialize theta and theta_0
    theta = np.zeros(shape=feature_matrix.shape[1])
    total_theta = np.zeros(shape=feature_matrix.shape[1])
    theta_0 = 0
    total_theta_0 = 0
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            result = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta = result[0]
            theta_0 = result[1]
            
            # sum and store theta/theta_0 values 
            total_theta += result[0]
            total_theta_0 += result[1]   
            
    # returns average 
    return (total_theta/(n*T), total_theta_0/(n*T))
    raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1:
        current_theta = (1- eta * L)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta * label
    else:
        current_theta = (1-eta*L) * current_theta
        current_theta_0 = current_theta_0

    return (current_theta, current_theta_0)        
    raise NotImplementedError
    """provided answer"""
    # mult = 1 - (eta * L)
    # if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
    #     return ((mult * current_theta) + (eta * label * feature_vector),
    #             (current_theta_0) + (eta * label))
    # return (mult * current_theta, current_theta_0)
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    theta = np.zeros(shape=feature_matrix.shape[1])
    theta_0 = 0
    count = 0

    for t in range(T):
        
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            count += 1
            T = 1 / np.sqrt(count)
            result = pegasos_single_step_update(feature_matrix[i], labels[i], L, T, theta, theta_0)
            
            # use the updated theta/theta_0 for next iter
            theta = result[0]
            theta_0 = result[1]
            

    return (theta, theta_0)
    raise NotImplementedError
    """ provided answer """
    # (nsamples, nfeatures) = feature_matrix.shape
    # theta = np.zeros(nfeatures)
    # theta_0 = 0
    # count = 0
    # for t in range(T):
    #     for i in get_order(nsamples):
    #         count += 1
    #         eta = 1.0 / np.sqrt(count)
    #         (theta, theta_0) = pegasos_single_step_update(
    #             feature_matrix[i], labels[i], L, eta, theta, theta_0)
    # return (theta, theta_0)
#pragma: coderesponse end

# feature_matrix = [
#     [ 0.1837462, 0.29989789, -0.35889786, -0.30780561, -0.44230703, -0.03043835, 0.21370063,  0.33344998, -0.40850817, -0.13105809],
#     [ 0.08254096, 0.06012654,  0.19821234, 0.40958367,  0.07155838, -0.49830717, 0.09098162,  0.19062183, -0.27312663,  0.39060785],
#     [-0.20112519, -0.00593087,  0.05738862, 0.16811148, -0.10466314, -0.21348009, 0.45806193, -0.27659307,  0.2901038,  -0.29736505],
#     [-0.14703536, -0.45573697, -0.47563745, -0.08546162, -0.08562345,  0.07636098, -0.42087389, -0.16322197, -0.02759763,  0.0297091],
#     [-0.18082261, 0.28644149, -0.47549449, -0.3049562, 0.13967768, 0.34904474, 0.20627692  0.28407868  0.21849356, -0.01642202]
#     ]


# Part II


#pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    num_sample = feature_matrix.shape[0]
    
    pred_arr = np.zeros(num_sample)
    for i in range(feature_matrix.shape[0]):       
        y = np.dot(theta, feature_matrix[i]) + theta_0
        if y > 0:
            pred_arr[i] = 1
        else:
            pred_arr[i] = -1
    return np.array(pred_arr)
    raise NotImplementedError
    """provided answer"""
    
    """answer 1"""
    # (nsamples, nfeatures) = feature_matrix.shape
    # predictions = np.zeros(nsamples)
    # for i in range(nsamples):
    #     feature_vector = feature_matrix[i]
    #     prediction = np.dot(theta, feature_vector) + theta_0
    #     if (prediction > 0):
    #         predictions[i] = 1
    #     else:
    #         predictions[i] = -1
    # return predictions

    """ answer 2
    Here, we use the fact that a boolean will be implicitly 
    casted by NumPy into 0 or 1 when mutiplied by a float.
    
    """
    # return (feature_matrix @ theta + theta_0 > 1e-7) * 2.0 - 1 
#pragma: coderesponse end


#pragma: coderesponse template
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
   
    preds_train = classify(train_feature_matrix, theta, theta_0)
    preds = classify(val_feature_matrix, theta, theta_0)
   
    train_accuracy = accuracy(preds_train, train_labels)
    test_accuracy = accuracy(preds, val_labels)
  
    return (train_accuracy, test_accuracy)

    raise NotImplementedError
    """ provided answer """
    # theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    # train_predictions = classify(train_feature_matrix, theta, theta_0)
    # val_predictions = classify(val_feature_matrix, theta, theta_0)
    # train_accuracy = accuracy(train_predictions, train_labels)
    # validation_accuracy = accuracy(val_predictions, val_labels)
    # return (train_accuracy, validation_accuracy)
#pragma: coderesponse end


#pragma: coderesponse template
def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    stop_words = []
    with open('stopwords.txt') as f:
        for line in f:
            stop_words.extend(line.split())

    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stop_words:
                dictionary[word] = len(dictionary)
    return dictionary
#pragma: coderesponse end





#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    
    num_reviews = len(reviews)
    print(num_reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    print(feature_matrix)
    return feature_matrix
#pragma: coderesponse end


#pragma: coderesponse template
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
#pragma: coderesponse end
