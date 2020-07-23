# -*- coding: utf-8 -*-

import numpy as np
import h5py
from Deep_neural_network import L_layer_model, predict
"""
This demonstrates the model building by using the numpy implementation of deep
neural network. A cat vs non-cat data set is used to train the neural network
for binary classfication.

"""

def load_preprocess_data(data_path):
    """
    loads the image data, flattens the image matrix to vector and standardize
    it.

    Parameters
    ----------
    data_path : string

    Returns
    -------
    train_x : numpy array
        shape = (no. of pixels, no. of training examples)
    train_y : numpy array
        shape = (1, no. of training examples)
    test_x : numpy array
        shape = (no. of pixels, no. of test examples)
    test_y : numpy array
        shape = (1, no. of test examples)
    classes : numpy array
        cat, non-cat

    """

    train_dataset = h5py.File(data_path + 'train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # train set features
    train_y = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File(data_path + 'test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # test set features
    test_y = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
    
    # flatten the image data.
    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T
    
    # standardize the image data
    train_x = train_x/255
    test_x = test_x/255
    
    return train_x, train_y, test_x, test_y, classes



data_path = './data/'
train_x, train_y, test_x, test_y, classes = load_preprocess_data(data_path)

# building a neural network with 4 layers (3 hidden). The input layer size is
# equal to train_x.shape[0]. First, second and third hidden layers have 20, 7,
# and 5 relu units, respectively. The output layer contains 1 sigmoid unit.
 
layers_dims = [train_x.shape[0], 20, 7, 5, 1]

parameters = L_layer_model(train_x, train_y, layers_dims = layers_dims, 
                            epochs = 1200, batch_size = 64, initialization = 'he',
                            optimizer = 'gd_with_momentum')

train_pred, train_acc = predict(train_x, train_y, parameters = parameters)
                                
test_pred, test_acc = predict(test_x, test_y, parameters = parameters)

print('Train accuracy: {}\nTest accuracy: {}'.format(train_acc, test_acc))
                              