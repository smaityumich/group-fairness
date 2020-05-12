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
    def __init__(self, architecture = [10, 20, 2], activation = tf.nn.relu, input_shape = (40,)):
        super(NNGraph, self).__init__()
        self.Layers = []
        if len(architecture) == 1:
            self.Layers.append(keras.layers.Dense(architecture[0], input_shape = input_shape, name = f'logit-layer'))

        elif len(architecture) == 2: 
            self.Layers.append(keras.layers.Dense(architecture[0], activation = activation, \
                    input_shape = input_shape, name = f'hidden-layer-1'))
            self.Layers.append(keras.layers.Dense(architecture[1], name = 'logit-layer'))

        elif len(architecture) == 3:
            self.Layers.append(keras.layers.Dense(architecture[0], activation = activation, \
                    input_shape = input_shape, name = f'hidden-layer-1'))
            self.Layers.append(keras.layers.Dense(architecture[1], activation = activation, \
                    input_shape = input_shape, name = f'hidden-layer-2'))
            self.Layers.append(keras.layers.Dense(architecture[2], name = 'logit-layer'))

        else:
            self.Layers.append(keras.layers.Dense(architecture[0], activation = activation, \
                    input_shape = input_shape, name = f'hidden-layer-1'))
            for layer_index, neurons in enumerate(architecture[1:-1], 2):
                    self.Layers.append(keras.layers.Dense(neurons, activation = activation, \
                                name = f'hidden-layer-{layer_index}'))
            self.Layers.append(keras.layers.Dense(architecture[-1], name = 'logit-layer'))

        
      
        self.model = keras.models.Sequential(self.layers)

    # Set forward pass
    def call(self, x, probability = False):
        out = self.model(x)

        if probability:
            out = tf.nn.softmax(out)
            out, _ = tf.linalg.normalize(out, ord = 1, axis = 1) # normalizes softmax vector to get probabilities 
        
        return out




