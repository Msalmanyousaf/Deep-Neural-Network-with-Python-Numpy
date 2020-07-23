# Implementation of Deep Neural Network using Python Numpy  
In this project, different concepts taught by Prof. Andrew Ng in the first two courses of [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) are implemented in Python Numpy to construct a deep neural network for binary classification based on user specified settings. The use of this neural network is then demonstrated by training a binary classifier to differentiate between cat and non-cat pictures.   
The following files/directories are attached:  
- **Deep_neural_network.py:** Containts all the functions required to build a deep neural network, train the parameters and then make predictions by using the trained parameters.  
- **data directory:** It contains *train_catvnoncat.h5* and *test_catvnoncat.h5* files having RGB image data of cats and non-cats objects. 
- **Cat_vs_non_cat_application.py:** Builds a deep neural network for binary classification of cat vs. non-cat images. This also includes a data preprocessing function which reads the image data from h5 files, flattens it and then standardize it to be fed into the neural network.   
## 1. Deep Neurual Network implementation  
The implemented neural network model expects the train_X of shape *(no. of features, no. of training examples)* and train_y of shape *(1, no. of training examples)*. train_y should contain 0/1 entries for binary classification. The user has also the freedom to provide the following settings:  
- **layers_dims:** It is a list which specifies how many layers you want to have and what the number of units in each layer are. A typical example for 4 layer nueral network might be: layers_dims = [no. of input features, L1 units, L2 units, L3 units, 1]. The final layer always has 1 unit because this implementation is intended for binary classification at the moment. 
- **learning_rate:** Learning rate for the optimizer.
- **activation:** Type of actiavtion to use for the hidden layers. The options are 'relu', 'leakyrelu', 'tanh' and 'sigmoid'. The output layer uses the 'sigmoid' unit.
- **epochs:** Number of full passes through the complete data set during the training.
- **initialization:** Type of initialization to use for the weights. The options are 'he' and 'xavier'. By default, initialization = 'he'
- **lambd:** Regularization parameter for L2 regularization. For the case where no regularization is used, lambd = 0. Regularization increases by increasing the lambd. By default, lambd = 0.
- **keep_prob:** This parameter allows the user to set the probability to keep the units output values if he wants to have Dropout. For no Dropout, keep_prob = 1 (keep all the units). In theory, L2 regularization can be used in combination with Dropout, but this implementation works only if the user wants to have either L2 regularization or Dropout, but not the both. This will be extended in future. By default, keep_prob = 1.
- **batch_size:** batch size to use mini-batch version of the optimization algorithm. By default, batch_size = None and it considers all the training examples in one single batch.
- **optimizer:** Type of optimizer to use. The options are 'gd', 'gd_with_momentum' and 'adam' for gradient descent, gradient descent with momentum and adam optimizer. By default, optimizer = 'gd'.
- **beta1:** The parameter to use for weighted moving average in gradient descent with momentum optimizer. This is also used in adam optimizer for first moment estimates. By default, beta1 = 0.9.
- **beta2:** The parameter to use for second moment estimates in adam optimizer. The default is 0.999.
- **epsilon:** A small number which is used in adam optimizer to avoid division by zero. The default is 10<sup>-8</sup>.  
### 1.1 Convention   
- **W:** Weights of the neural network.
- **b:** Biases of the neural network units
- **Shape of W:** (no. of units in current layer , no. of units/examples in the previous layer)
- **Shape of b:** (no. of units in the current layer , 1)
- **l:** Layer number. Input layer is numbered as zero. The first hidden layer is layer 1. The output layer is L.
- **Z:** Output of units before applying the non-linearity. Z<sup>l</sup> = W<sup>l</sup>A<sup>l-1</sup> + b<sup>l</sup>
- **A:** Output of the units after applying non-linearity (e.g. Relu).
- **dA:** Derivative of cost w.r.t. A. The considered cost in this impelemtation is logistic loss.
- **dZ:** Derivative of cost w.r.t. Z.
- **dW:** Derivative of cost w.r.t. weights W.
- **db:** Derivative of cost w.r.t. biases b.
- **Note:** In the same layer, Z, A, dA and dZ have same shapes. Similarly, W and dW have same shapes. b and db have same shapes.
- **parameters:**: This is the dictionary containing the weights and the biases. The keys are: W1, b1, W2, b2, .... ,WL, bL
- **grads:** This is the dictionary containing the derivatives w.r.t. the cost. The keys are: 'dA' +str(l-1), 'dW' +str(l) and 'db' +str(l), where l is the layer number.
- **predict:** This is the function which is used to make predictions while using the trained weightes. The probalities greater than 0.5 are converted into 1s and the rest in 0s. Please provide the same activation ('relu' etc) here which you used during the training. By default, the activation is 'relu'.




