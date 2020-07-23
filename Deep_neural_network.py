# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)

    Parameters
    ----------
    X : numpy array
        input data of shape = (no. of features, no. of total examples)
    Y : numpy array
        ground truth labels for binary classification. 
        shape = (1, no. of total examples)
    mini_batch_size : int, optional
        No. of training examples in one mini-batch. The default is 64.

    Returns
    -------
    mini_batches : python list
        containing all the mini-batches
        mini_batches = [ (mini_batch_1_X, mini_batch_1_Y),
                        (mini_batch_2_X, mini_batch_2_Y), ... ]

    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    
    # number of mini batches of size mini_batch_size
    num_complete_minibatches = math.floor( m / mini_batch_size ) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : 
                                  k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : 
                                  k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy.

    Parameters
    ----------
    Z : numpy array of any shape
        output of a neuron before appying non-linearity.

    Returns
    -------
    A : numpy array of shape same as Z
        output of sigmoid(Z).
    cache : numpy array of shape same as Z
        Here, cache is the same as input Z. It contains information that might
        be useful during backpropagation.

    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z): 
    """
    Implements the RELU function using numpy.

    Parameters
    ----------
    Z : numpy array of any shape
        output of a neuron before appying non-linearity.

    Returns
    -------
    A : numpy array of shape same as Z
        output of relu(Z).
    cache : numpy array of shape same as Z
        Here, cache is the same as input Z. It contains information that might
        be useful during backpropagation.

    """
    A = np.maximum(0, Z)
    
    cache = Z 
    return A, cache

def tanh(Z):
    """
    Implements the tan-hyperbolic function using numpy.

    Parameters
    ----------
    Z : numpy array of any shape
        output of a neuron before appying non-linearity.

    Returns
    -------
    A : numpy array of shape same as Z
        output of tanh(Z).
    cache : numpy array of shape same as Z
        Here, cache is the same as input Z. It contains information that might
        be useful during backpropagation.

    """
    A = np.tanh(Z)
    cache = Z
    return A, cache

def leakyrelu(Z): 
    """
    Implements the Leaky RELU function using numpy.

    Parameters
    ----------
    Z : numpy array of any shape
        output of a neuron before appying non-linearity.

    Returns
    -------
    A : numpy array of shape same as Z
        output of leakyrelu(Z).
    cache : numpy array of shape same as Z
        Here, cache is the same as input Z. It contains information that might
        be useful during backpropagation.

    """
    
    # in this implementation, the slope of leakyrelu(Z) is assumed to be 0.01
    # for Z < 0
    A = np.maximum(0.01*Z, Z)

    cache = Z 
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single SIGMOID unit.

    Parameters
    ----------
    dA : numpy array
        Gradients of cost w.r.t. post-activation vales. It can have any shape.
    cache : numpy tuple
        cache = ( (A_prev, W, b), Z) without Dropout
        cache = ( (A_prev, W, b, D), Z) with Dropout
        This is stored while doing forward prop on layer l. See function 
        'forward_prop_one_layer'.

    Returns
    -------
    dZ : numpy array of same shape as dA and Z
        Gradient of the cost with respect to Z.

    """

    Z = cache[1]
    
    s = 1/(1+np.exp(-Z))   # sigmoid function formula
    ds = s * (1-s)   # derivative of sigmoid function
    dZ = dA * ds

    
    return dZ

def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single RELU unit.

    Parameters
    ----------
    dA : numpy array
        Gradients of cost w.r.t. post-activation vales. It can have any shape.
    cache : numpy tuple
        cache = ( (A_prev, W, b), Z) without Dropout
        cache = ( (A_prev, W, b, D), Z) with Dropout
        This is stored while doing forward prop on layer l. See function 
        'forward_prop_one_layer'.

    Returns
    -------
    dZ : numpy array of same shape as dA and Z
        Gradient of the cost with respect to Z.

    """

    Z = cache[1]
    dZ = np.array(dA, copy=True) # just converting dZ to a correct object.
    
    # When Z <= 0, set dZ to 0 as well. This is because the derivativative
    # of relu function becomes zero for Z <= 0.
    dZ[Z <= 0] = 0

    
    return dZ

def tanh_backward(dA, cache):
    """
    Implements the backward propagation for a single tan-hyperbolic unit.

    Parameters
    ----------
    dA : numpy array
        Gradients of cost w.r.t. post-activation vales. It can have any shape.
    cache : numpy tuple
        cache = ( (A_prev, W, b), Z) without Dropout
        cache = ( (A_prev, W, b, D), Z) with Dropout
        This is stored while doing forward prop on layer l. See function 
        'forward_prop_one_layer'.

    Returns
    -------
    dZ : numpy array of same shape as dA and Z
        Gradient of the cost with respect to Z.

    """

    Z = cache[1]
    
    s = np.tanh(Z) # tan-hyperbolic activation
    
    ds = 1 - np.square(s) # derivative of sigmoid function
    dZ = dA * ds

    
    return dZ

def leakyrelu_backward(dA, cache):
    """
    Implements the backward propagation for a single LEAKY RELU unit.

    Parameters
    ----------
    dA : numpy array
        Gradients of cost w.r.t. post-activation vales. It can have any shape.
    cache : numpy tuple
        cache = ( (A_prev, W, b), Z) without Dropout
        cache = ( (A_prev, W, b, D), Z) with Dropout
        This is stored while doing forward prop on layer l. See function 
        'forward_prop_one_layer'.

    Returns
    -------
    dZ : numpy array of same shape as dA and Z
        Gradient of the cost with respect to Z.

    """

    Z = cache[1]
    dZ = np.array(dA, copy=True) # just converting dZ to a correct object.
    
    # assuming leaky relu with slope of 0.01 when Z <=0.
    dZ[Z <= 0] *= 0.01

    
    return dZ

def initialize_parameters(layer_dims, initialization):
    """
    Initializes the weights and biases for the neural network.

    Parameters
    ----------
    layer_dims : python list
        Contains the dimensions of each layer of the neural network (including
        the input layer)
    initialization : string
        Type of initialization to use for the weights. The options are 'he' and
        'xavier'.     

    Returns
    -------
    parameters : python dictionary
        Contains the initialized weights and biases parameters. The keys of the 
        dictionary are W1, b1,...,WL, bL
        Wl -- weight matrix of layer l having shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of layer l having shape (layer_dims[l], 1)

    """

    parameters = {}
    L = len(layer_dims)    # number of layers in the network, including the input layer
    
    
    if initialization == 'he':

        for l in range(1, L):  # starting from 1 because input layer is labelled as 0.
            parameters['W' + str(l)] = np.random.randn( layer_dims[l], layer_dims[l-1]
                                                       ) * np.sqrt(2 / layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
            
    elif initialization == 'xavier':

        for l in range(1, L):  # starting from 1 because input layer is labelled as 0.
            parameters['W' + str(l)] = np.random.randn( layer_dims[l], layer_dims[l-1]
                                                       ) * np.sqrt(1 / layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
            
    return parameters

def forward_prop_one_layer (A_prev, W, b, activation, keep_prob = 1.0):
    """
    Given the weights and biases of the current layer and the activations from
    the previous layer, it performs the forward propagation on the current layer
    according to the specified activation type.

    Parameters
    ----------
    A_prev : numpy array
        activations from previous layer (or input data)
        shape = (size of previous layer, number of training examples)
    W : numpy array
        weights matrix for the current layer 
        shape = (size of current layer, size of previous layer)
    b : numpy array
        bias vector. shape = (size of the current layer, 1)
    activation : string
        the activation to be used in this layer. Options are: "sigmoid", "relu",
        "leakyrelu" or "tanh".
    keep_prob : float, optional
        Probability value to keep the neurons to implement Dropout. The default 
        is 1.0 (keep all the neurons, i.e. no Dropout)

    Returns
    -------
    A : numpy array
        the output after applying activation function on the current layer.
        shape = (size of the current layer, number of training examples)
    cache : python tuple
        No Dropout: cache = ( (A_prev, W, b), Z ), where Z = W*A_prev + b
        With Dropout: cache = ( (A_prev, W, b, D), Z ), where Z = W*A_prev + b,
        and D is the matrix containing 1s for the neurons which survived after 
        Dropout application.

    """

    Z = np.dot(W, A_prev) + b
    
    if activation == "sigmoid":
        A, cache_2 = sigmoid(Z) 
        
    elif activation == "relu":
        A, cache_2 = relu(Z)     
        
    elif activation == "leakyrelu":
        A, cache_2 = leakyrelu(Z)
        
    elif activation == "tanh":
        A, cache_2 = tanh(Z)
    
    # apply Dropout
    if keep_prob < 1.0:
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)    # create a mask of 0s and 1s
        A = np.multiply(D, A)
        A /= keep_prob
        cache_1 = (A_prev, W, b, D)
    elif keep_prob == 1:
        cache_1 = (A_prev, W, b)
    
    cache = (cache_1, cache_2)
    return A, cache


def forward_prop_all_layers (X, parameters, activation, keep_prob = 1.0):
    """
    This function performs forward propagation on all the hidden layers based on
    the specified activation type. Then, it performs the forward propagation for
    the output layer based on 'sigmoid' activation. Therefore, this neural network
    is designed for binary classification problems.

    Parameters
    ----------
    X : numpy array of shape (no. of input features, no. of training examples)
        Input data matrix.
    parameters : python dictionary
        This contains the weights and biases for all the layers.
        The keys of the dictionary can be found in function "initialize_parameters".
    activation : string
        The activation to be used for the hidden layer. Options are: "sigmoid", 
        "relu", "leakyrelu" or "tanh".
    keep_prob : float, optional
        Probability value to keep the neurons to implement Dropout. The default 
        is 1.0 (keep all the neurons, i.e. no Dropout)

    Returns
    -------
    AL : numpy array
        Post-activtion value of the output layer. shape = (1, no. of training examples)
    caches : python list
        caches = [ cache_L1, cache_L2,...,cache_LL ]
        where cache_L1 = cache of layer 1. This is returned by the function 
        'forward_prop_one_layer', when applied to layer 1.
        cache_LL = cache of output layer.

    """
 
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network (without input layer)

    # loop over all hidden layers. Input layer is the 0th layer
    for l in range(1, L):
        A_prev = A 
        # forward-prop on the current layer
        A, cache = forward_prop_one_layer (A_prev, parameters['W' + str(l)], 
                                           parameters['b' + str(l)], activation,
                                           keep_prob)
        caches.append(cache)
    
    
    # Implement the sigmoid activation for the output layer. This neural network
    # is designed for binary classification problems. 
    AL, cache = forward_prop_one_layer (A, parameters['W' + str(L)], 
                                        parameters['b' + str(L)], 'sigmoid')
    # although Dropout is not applied to the output layer, but to be consistent,
    # save some garbage value as the dropout mask also in the cache of the output
    # layer. This will simplify things while implementing backpropagation with dropout.
    
    if keep_prob < 1:    # meaning, the user wants to apply Dropout
        cache_temp_0 = list(cache[0])
        cache_temp_1 = cache[1]   
        cache_temp_0.append(0)  # added 0 as Dropout mask for the output layer
        cache = (tuple(cache_temp_0), cache_temp_1)
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y, lambd = 0.0, W_fr = 0.0):
    """
    Computes the total cross-entropy loss (log loss) of the predictions w.r.t. the 
    ground truth values. cost returned by this function is total cost of all the 
    training examples. So, divide it by the training examples afterwards.

    Parameters
    ----------
    AL : numpy array
        Post-activtion value of the output layer. shape = (1, no. of training examples)
    Y : numpy array
        true "label" vector (for example: containing 0 if non-cat, 1 if cat).
        shape = (1, no. of training examples)
    lambd : float, optional
        Regularization parameter lambda for L2 regularization. The default is 
        0 (no regularization).
    W_fr : float, optional
        sum of squared Frobenius norm of the weights of all layers. This is used
        in L2 regularization. The default is 0.0.

    Returns
    -------
    cost : float
        total cross-entropy cost

    """

    AL_logs = [np.log(AL), np.log(1-AL)]
    
    # compute the unregularized part of the cost. Don't divide it by number of
    # training examples yet.
    cost = ( -np.dot(Y, AL_logs[0].T) - np.dot((1-Y), AL_logs[1].T))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    if lambd != 0: # take L2 regularization into account
        # compute the regularized part of the cost. Don't divide it by number of
        # training examples yet.
        L2_reg_cost = ( lambd * W_fr ) / 2
        cost += L2_reg_cost
    
  
    return cost


def back_prob_one_layer (dA, cache, activation, lambd = 0.0, keep_prob = 1.0):
    """
    Given the derivatives of the cost w.r.t. the activations of the current layer,
    this function performs the backpropagation step for the current layer according
    to the specified activation type.

    Parameters
    ----------
    dA : numpy array
        gradients of the cost w.r.t. the post activation values of the current
        layer. It has same shape as A.
    cache : python tuple
        without Dropout: cache = ( (A_prev, W, b), Z ).
        with Dropout: cache = ( (A_prev, W, b, D), Z ).
        It is returned by the function 'forward_prop_one_layer'.
    activation : string
        The activation to be used. Options are: "sigmoid", "relu", "leakyrelu" 
        or "tanh".
    lambd : float, optional
        Regularization parameter for L2 regularization. The default is 0 (no regularization)
    keep_prob : float, optional
        probability values to implement Dropout. The default is 1.0 (no Droput).

    Returns
    -------
    dA_prev : numpy array
        gradients of the cost w.r.t. the post activation values of the previous
        layer. It has same shape as A_prev.
    dW : numpy array
        Gradients of the cost with respect to W (current layer l), same shape as W.
    db : numpy array
        Gradient of the cost with respect to b (current layer l), same shape as b.

    """

    m = dA.shape[1]   # total training examples in the current batch
    if keep_prob < 1:
        A_prev, W, b, _ = cache[0]
    elif keep_prob == 1:
        A_prev, W, b = cache[0]
    
    if activation == "relu":
        dZ = relu_backward(dA, cache)
        dW = np.dot(dZ, A_prev.T)/m + lambd * W / m
        db = np.sum(dZ, axis = 1, keepdims = True)/m
        dA_prev = np.dot(W.T, dZ)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache)
        dW = np.dot(dZ, A_prev.T)/m + lambd * W / m
        db = np.sum(dZ, axis = 1, keepdims = True)/m
        dA_prev = np.dot(W.T, dZ)
        
    elif activation == "tanh":
        dZ = tanh_backward(dA, cache)
        dW = np.dot(dZ, A_prev.T)/m + lambd * W / m
        db = np.sum(dZ, axis = 1, keepdims = True)/m
        dA_prev = np.dot(W.T, dZ)
        
    elif activation == "leakyrelu":
        dZ = leakyrelu_backward(dA, cache)
        dW = np.dot(dZ, A_prev.T)/m + lambd * W / m
        db = np.sum(dZ, axis = 1, keepdims = True)/m
        dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def back_prop_all_layers (AL, Y, caches, activation, lambd = 0.0, keep_prob = 1.0):
    """
    Implements the backpropagation for all the layers of neural network.

    Parameters
    ----------
    AL : numpy array
        Activations of the output sigmoid unit. shape = (1, no. of training examples)
    Y : numpy array
        true "label" vector (for example: containing 0 if non-cat, 1 if cat).
        shape = (1, no. of training examples)
    caches : python list
        It is returned by the function 'forward_prop_all_layers'
    activation : string
        The activation to be used for the hidden layers. Options are: "sigmoid", 
        "relu", "leakyrelu" or "tanh".
    lambd : float, optional
        Regularization parameter for L2 regularization. The default is 0 (no regularization)
    keep_prob : float, optional
        probability values to implement Dropout. The default is 1.0 (no Droput).

    Returns
    -------
    grads : A dictionary with the gradients
        grads["dA" + str(l-1)] = ...
        grads["dW" + str(l)] = ...
        grads["db" + str(l)] = ...

    """
    grads = {}
    L = len(caches) # the number of layers (without the input layer)
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation by finding the derivatives of the cost 
    # function w.r.t. the post activation values of the output layer. Remember, 
    # the output layer consists of single sigmoid neuron. The cost function is
    # log loss. 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1] # cache of the output layer. The index of caches starts at zero.
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = back_prob_one_layer(
                                                dAL, current_cache, 'sigmoid',
                                                lambd, keep_prob)
    if keep_prob < 1: # apply Dropout
        previous_cache = caches[L-2]
        _,_,_,D = previous_cache[0]
        grads["dA" + str(L-1)] = np.multiply(grads["dA" + str(L-1)], D)
        grads["dA" + str(L-1)] /= keep_prob
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
 
        current_cache = caches[l]   # current hidden layer cache
        dA_prev_temp, dW_temp, db_temp = back_prob_one_layer(
                                    grads['dA'+str(l+1)], current_cache, 'relu',
                                    lambd, keep_prob)
        
        # current hidden layer weight/bias derivatives
        grads["dW" + str(l + 1)] = dW_temp 
        grads["db" + str(l + 1)] = db_temp
        
        if keep_prob < 1 and l > 0: # apply Dropout to the hidden layers
            previous_cache = caches[l-1]
            _,_,_,D = previous_cache[0]
            dA_prev_temp = np.multiply(dA_prev_temp, D)
            dA_prev_temp /= keep_prob
            
        # previous hidden layer activation derivatives
        grads["dA" + str(l)] = dA_prev_temp

    return grads

def initialize_gd_with_momentum(parameters):
    """
    Initializes the velocity to be used in gradient descent with momentum.

    Parameters
    ----------
    parameters : python dictionary containing neural network parameters
        parameters = { 'W1': ..., 'b1': ..., 'W2': ..., 'b2': ...,...
                      'WL': ...,'bL': ...}

    Returns
    -------
    v : python dictionary
        v = { 'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...,...'dWL': ...,'dbL': ...}

    """

    L = len(parameters) // 2 # number of layers in the neural networks (without input layer)
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros( (parameters['W'+str(l+1)].shape[0], 
                                        parameters['W'+str(l+1)].shape[1]) )
        
        v["db" + str(l+1)] = np.zeros( (parameters['b'+str(l+1)].shape[0], 
                                        parameters['b'+str(l+1)].shape[1]) )
        
    return v

def initialize_adam(parameters):
    """
    Initializes the variables v and s which are used in adam optimizer to 
    perform weighted moving average.

    Parameters
    ----------
    parameters : python dictionary containing neural network parameters
        parameters = { 'W1': ..., 'b1': ..., 'W2': ..., 'b2': ...,...
                      'WL': ...,'bL': ...}

    Returns
    -------
    v : python dictionary
        v = { 'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...,...'dWL': ...,'dbL': ...}
    s : python dictionary
        s = { 'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...,...'dWL': ...,'dbL': ...}

    """

    L = len(parameters) // 2 # number of layers in the neural networks (without input layer)
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros( (parameters['W'+str(l+1)].shape[0], 
                                        parameters['W'+str(l+1)].shape[1]) )
        
        v["db" + str(l+1)] = np.zeros( (parameters['b'+str(l+1)].shape[0], 
                                        parameters['b'+str(l+1)].shape[1]) )
        
        s["dW" + str(l+1)] = np.zeros( (parameters['W'+str(l+1)].shape[0], 
                                        parameters['W'+str(l+1)].shape[1]) )
        
        s["db" + str(l+1)] = np.zeros( (parameters['b'+str(l+1)].shape[0], 
                                        parameters['b'+str(l+1)].shape[1]) )
    
    return v, s

def update_parameters_gd(parameters, grads, learning_rate):
    """
    Updates the neural network parameters using gradient descent algorithm.

    Parameters
    ----------
    parameters : python dictionary containing the parameter values
        {W1: ...,b1: ..., W2: ..., b2: ...,...,WL: ...,bL: ...}
    grads : Python dictionary containing the gradients.
        It is returned by the function 'back_prop_all_layers'.
    learning_rate : Float

    Returns
    -------
    parameters : Python dictionary
        It contains the updated parameter values.

    """

    L = len(parameters) // 2 # number of layers in the neural network (without the input layer)
    
    # updating parameters using gradient descent algorithm
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        
    return parameters


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update the parameters using gradient descent with momentum

    Parameters
    ----------
    parameters : python dictionary containing the parameter values
        {W1: ...,b1: ..., W2: ..., b2: ...,...,WL: ...,bL: ...}
    grads : Python dictionary containing the gradients.
        It is returned by the function 'back_prop_all_layers'.
    v : python dictionary
        v = { 'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...,...'dWL': ...,'dbL': ...}
    beta : float
    learning_rate : float

    Returns
    -------
    parameters : Python dictionary
        It contains the updated parameter values.
    v : Python dictionary
        It contains the updated velocities for gradient descent with momentum.

    """

    L = len(parameters) // 2 # number of layers in the neural networks (without input layer)
    
    # Momentum update for each parameter
    for l in range(L):
        
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta)*grads['dW'+ str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta)*grads['db'+ str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * v["db" + str(l+1)]

        
    return parameters, v


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Updates parameters using adam optimizer.

    Parameters
    ----------
    parameters : python dictionary containing the parameter values
        {W1: ...,b1: ..., W2: ..., b2: ...,...,WL: ...,bL: ...}
    grads : Python dictionary containing the gradients.
        It is returned by the function 'back_prop_all_layers'.
    v : python dictionary
        Adam variable, moving average of the first gradient
        v = { 'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...,...'dWL': ...,'dbL': ...}
    s : python dictionary
        Adam variable, moving average of the squared gradient.
        s = { 'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...,...'dWL': ...,'dbL': ...}
    t : int
        iteration number
    learning_rate : float, optional
    beta1 : float, optional
        Exponential decay hyperparameter for the first moment estimates.
        The default is 0.9.
    beta2 : float, optional
        Exponential decay hyperparameter for the second moment estimates. 
        The default is 0.999.
    epsilon : float, optional
        paramter used in adam optimizer to avoid division by zero.

    Returns
    -------
    parameters : Python dictionary
        It contains the updated parameter values.
    v : python dictionary
        Updated Adam variable, moving average of the first gradient
    s : python dictionary
        Updated Adam variable, moving average of the squared gradient.

    """

    L = len(parameters) // 2  # number of layers in the neural networks (without the input layer)
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
 
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. 
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-(beta1)**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-(beta1)**t)
 
        # Moving average of the squared gradients. 

        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*np.square(grads["db" + str(l+1)])

        # Compute bias-corrected second raw moment estimate. 
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-(beta2)**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-(beta2)**t)

        # Update parameters. 
        parameters["W" + str(l+1)] -= learning_rate*(
            v_corrected["dW" + str(l+1)])/(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        
        parameters["b" + str(l+1)] -= learning_rate*(
            v_corrected["db" + str(l+1)])/(np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    return parameters, v, s


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, activation = 'relu',
                  epochs = 3000, print_cost = True, initialization = 'he',
                  lambd = 0.0, keep_prob = 1.0, batch_size = None, beta1 = 0.9,
                  beta2 = 0.999, epsilon = 1e-8, optimizer = 'gd'):
    """
    Implements an L-layered neural network architecture for binary classification.

    Parameters
    ----------
    X : numpy array
        shape = (no. of features, no. of training examples)
    Y : numpy array
        shape = (1, no. of training examples)
    layers_dims : list
        Contains the dimensions of each layer (including the input layer)
    learning_rate : float, optional
        learning rate for gradient descent. The default is 0.0075.
    activation : string, optional
        Activation type ('relu', 'sigmoid', 'leakyrelu', 'tanh') for the hidden 
        layers. The default is 'relu'.
    epochs : int, optional
        Total number of times we need to pass through whole training set while
        optimizing. The default is 3000.
    print_cost : logical, optional
        To print training cost after every 100 epochs. The default is True.
    initialization : string, optional
        Type of initialization to use for the weights. The options are 'he' and
        'xavier'. The default is 'he'.
    lambd : float, optional
        Regularization parameter lambda for L2 regularization. The default is
        0.0 (no regularization).
    keep_prob : float, optional
        Probability value to keep the neurons to implement Dropout. The default 
        is 1.0 (keep all the neurons, i.e. no Dropout)
    batch_size : int, optional
        batch size to consider while optimizing. The default is None (all the 
        training examples are considered at the same time.)
    beta1 : float, optional
        parameter which is used for weighted moving average in gradient descent
        with momentum optimizer or adam optimizer. The default is 0.9.
    beta2 : float, optional
        parameter which is used for weighted moving average in adam optimizer. 
        The default is 0.999.
    epsilon : float, optional
        paramter used in adam optimizer to avoid division by zero.
    optimizer : string, optional
        type of optimizer to use. The options are 'gd' for gradient descent,
        'gd_with_momentum' for gradient descent with momentum and 'adam' for
        adam optimizer. The default is 'gd'.
        
    Returns
    -------
    parameters : python dictionary
        trained neural network parameters.
        {W1: ...,b1: ..., W2: ..., b2: ...,...,WL: ...,bL: ...}

    """
    
    m = X.shape[1]             # number of training examples
    # if no batch size is provided, consider all the training examples as one 
    # batch. 
    if not batch_size: 
        batch_size = m
        
    # in practice, both Dropout and L2 regularization can be applied but this
    # implementation handles only one at a time. So, either apply Dropout or 
    # L2 regularization.
    assert (lambd == 0 or keep_prob == 1)
    
    costs = []          # keep track of cost
    total_layers = len(layers_dims)  # including the input layer
    t = 0         # initializing the counter required for Adam update
    
    # Parameters initialization. 
    parameters = initialize_parameters(layers_dims, initialization)
    
    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "gd_with_momentum":
        v = initialize_gd_with_momentum(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
   
    # Loop (optimization)
    for i in range(0, epochs):
               
        minibatches = random_mini_batches(X, Y, batch_size)
        cost = 0
        
        for minibatch in minibatches:   # fit all the mini-batches
            
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = forward_prop_all_layers (minibatch_X, parameters, 
                                                  activation, keep_prob)
                
            W_fr = 0 # to collect the square of Frobenius norm for all the weights
            
            if lambd != 0:  # check whether the user wants L2 regularization
                for ll in range (1, total_layers):
                    W_fr += np.sum( np.square(parameters['W' + str(ll)]) )
                
            # Compute cost.
            cost += compute_cost(AL, minibatch_Y, lambd, W_fr)
                
            # Backward propagation without Dropout
            grads = back_prop_all_layers (AL, minibatch_Y, caches, 
                                          activation, lambd, keep_prob)
    
            # Update parameters.
            if optimizer == "gd":
                parameters = update_parameters_gd (parameters, grads, learning_rate)
            elif optimizer == "gd_with_momentum":
                parameters, v = update_parameters_with_momentum(parameters, 
                                                                grads, v, beta1, 
                                                                learning_rate)
            elif optimizer == "adam":
                t = t + 1    # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, 
                                                               v, s,t, learning_rate, 
                                                               beta1, beta2, epsilon)
        
        cost /= m
               
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
            costs.append(cost)
        # if user does not want to see the cost, at least show the epoch number after
        # every 10 epochs.
        if not print_cost:
            if i % 10 == 0:
                print('epoch: {}'.format(i))
            
    # plot the cost
    if print_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        

    
    return parameters

def predict(X, y, parameters, activation = 'relu'):
    """
    Returns the predicted labels for the binary classification by using
    the trained neural network weights. It also prints the accuracy.

    Parameters
    ----------
    X : numpy array
        shape = (no. of features, no. of examples)
    y : numpy array
        True labels, shape = (1, no. of examples)
    parameters : python dictionary
        trained neural network parameters.
        {W1: ...,b1: ..., W2: ..., b2: ...,...,WL: ...,bL: ...}.
    activation : activation, optional
        Activation used for the hidden layers while training.
        The default is 'relu'.

    Returns
    -------
    p : int
        Predicted label. 0,1 label for the binary classification
    accuracy : float
        accuracy of binary classification.

    """

    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = forward_prop_all_layers (X, parameters, activation)

    
    # convert probabilities to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
  
    accuracy = np.sum((p == y)/m)
        
    return p, accuracy
