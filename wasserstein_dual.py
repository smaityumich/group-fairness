import tensorflow as tf
import nn_graph

def wasserstein_distance(x, y, epoch = 200, wasserstein_lr = 1e-4, epsilon = 1e0, \
    architecture = [10, 1], activation = tf.nn.relu, input_shape = (2,)):

    u = nn_graph.NNGraph(architecture=architecture, activation=activation, input_shape=input_shape, name = 'potential-1')
    v = nn_graph.NNGraph(architecture=architecture, activation=activation, input_shape=input_shape, name = 'potential-2')

    optimizer = tf.optimizers.Adam(wasserstein_lr)

    # Incomplete code