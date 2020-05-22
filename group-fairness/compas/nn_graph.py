import tensorflow as tf
from tensorflow import keras


class NNGraph(keras.Model):

    """
    Class for neural net graph

    initialization: 

    architecture: (list of int) list of neuron numbers in each layers; one layer is equivalent to logistic regression
    activation: (Tensorflow Core function) activation function for hidden layers
    input_shape: (tuple) provide input shape; if vector provide (dimension, ) 


    variables:

    Layers: (list of layers from keras.layers API) contains all the layers
    model: (keras.model.Sequential object)


    """

    # Set layers
    def __init__(self, architecture = [10, 20, 2], activation = tf.nn.relu, input_shape = (40,), convex = False, name = None):
        super(NNGraph, self).__init__()
        self.Layers = []
        
        if convex:
            kernel_constraint = keras.constraints.NonNeg()
            kernel_initializer = tf.random_uniform_initializer(minval = 0, maxval = 0.1, seed = 1)
        else:
            kernel_constraint = None
            kernel_initializer = None

        if len(architecture) == 1:
            self.Layers.append(keras.layers.Dense(architecture[0], input_shape = input_shape,\
                 kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer, name = f'logit-layer'))

        elif len(architecture) == 2: 
            self.Layers.append(keras.layers.Dense(architecture[0], activation = activation, \
                    kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                         input_shape = input_shape, name = f'hidden-layer-1'))

            self.Layers.append(keras.layers.Dense(architecture[1], kernel_constraint = kernel_constraint,\
                 kernel_initializer = kernel_initializer, name = 'logit-layer'))


        elif len(architecture) == 3:
            self.Layers.append(keras.layers.Dense(architecture[0], activation = activation, \
                    kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                         input_shape = input_shape, name = f'hidden-layer-1'))

            self.Layers.append(keras.layers.Dense(architecture[1], activation = activation, \
                    kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                         input_shape = input_shape, name = f'hidden-layer-2'))

            self.Layers.append(keras.layers.Dense(architecture[2],\
                kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                     name = 'logit-layer'))


        else:
            self.Layers.append(keras.layers.Dense(architecture[0], activation = activation, \
                    kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                         input_shape = input_shape, name = f'hidden-layer-1'))

            for layer_index, neurons in enumerate(architecture[1:-1], 2):
                    self.Layers.append(keras.layers.Dense(neurons, activation = activation, \
                                kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                                     name = f'hidden-layer-{layer_index}'))

            self.Layers.append(keras.layers.Dense(architecture[-1],\
                 kernel_constraint = kernel_constraint, kernel_initializer = kernel_initializer,\
                      name = 'logit-layer'))

        
      
        self.model = keras.models.Sequential(self.layers, name = name)

    # Set forward pass
    def call(self, x, probability = False):
        out = self.model(x)

        if probability:
            out = tf.nn.softmax(out)
            out, _ = tf.linalg.normalize(out, ord = 1, axis = 1) # normalizes softmax vector to get probabilities 
        
        return out




