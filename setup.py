import tensorflow as tf
import numpy as np
import datetime
import nn_graph
import utils



class GroupFairness():

    def __init__(self, batch_size = 250, epoch = 1000, learning_rate = 1e-4, \
                        l2_regularizer = 0, clip_grad = 40, seed = 1):

        super(GroupFairness, self).__init__()
        self.batch_size = batch_size
        self.epoch = epoch
        self.l2_regularizer = l2_regularizer
        self.clip_grad = clip_grad
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.seed = seed

    def set_graph(self, classifier_architecture = [10, 20, 2], classifier_input_shape = (40, ),\
         potential_architecture = [10, 1], potential_input_shape = (2,), activation = tf.nn.relu):

        self.classifier = nn_graph.NNGraph(architecture=classifier_architecture, activation=activation, \
            input_shape=classifier_input_shape, name='classifier')
        self.potential_y0 = [nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label0-group0'), \
                nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label0-group1')]
        self.potential_y1 = [nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label1-group0'), \
                nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label1-group1')]


    def create_batch_data(self, data_train, data_test):
        # Tensor slices for train data
        batch = tf.data.Dataset.from_tensor_slices(data_train)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_train_data = batch.take(self.epoch)

        # Tensor slices for test data
        batch = tf.data.Dataset.from_tensor_slices(data_test)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_test_data = batch.take(self.epoch)

    def wasserstein_distance(self, x1, x2, potential1 = None, potential2 = None, \
         epsilon = 1e-1, reguralizer = 'entropy'):
        """
        Given dual potential functions calculates the regularized Wasserstein distance

        parameters:
        x1: tensor of dimension 2
        x2: tensor of dimension 2
        potential1: (tensor graph) dual potential for first data
        potential2: (tensor graph) dual potential for second data
        epsilon: (float) regularizing parameter
        regularizer: (string) method of regularization. Options: (1) entropy, and (2) L2

        return:
        distance: (tensor of shape ()) regularized wasserstein distance

        reference:
        [1] Seguy et. al.: 'Large-scale optimal transport and mapping estimation'
        """
        self.epsilon = epsilon
        if potential1 == None:
            potential1 = self.potential_y0[0]
        if potential2 == None:
            potential2 = self.potential_y0[1]
        u = potential1(x1)
        v = potential2(x2)
        x1_expanded = tf.tile(tf.expand_dims(x1, axis = 1), [1, x2.shape[0], 1]) # shape (nx1, nx2, 2)
        x2_expanded = tf.tile(tf.expand_dims(x2, axis = 0), [x1.shape[0], 1, 1]) # shape (nx1, nx2, 2)
        pairwise_distance = 0.5 * tf.reduce_sum((x1_expanded - x2_expanded)**2, axis = 2) # shape (nx1, nx2)
        u_expanded = tf.tile(tf.expand_dims(u, axis = 1), [1, v.shape[0], 1]) # shape (nu, nv, 1)
        v_expanded = tf.tile(tf.expand_dims(v, axis = 0), [u.shape[0], 1, 1]) # shape (nu, nv, 1)
        pairwise_distance = tf.reshape(pairwise_distance, (pairwise_distance.shape[0], pairwise_distance.shape[1], 1))
        L = u_expanded + v_expanded - pairwise_distance
        if reguralizer == 'entropy':
            penalty = -epsilon * tf.exp((1/epsilon) * L)
        elif reguralizer == 'L2':
            penalty = -(1/(4*epsilon))*(tf.nn.relu(L)**2)
        else:
            raise TypeError('Wrong entry in regularizer. Options: entropy and L2')
        distance = tf.reduce_mean(u) + tf.reduce_mean(v) + tf.reduce_mean(penalty)
        return distance


    def create_tensorboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        expt_id = np.random.randint(1000000)
        parameter = f'-expt_id-{expt_id}lr-{self.learning_rate}-w_reg-{self.epsilon}-l2_reg-{self.l2_regularizer}'
        train_log_dir = 'logs/' + current_time + parameter + '/train'
        test_log_dir = 'logs/' + current_time + parameter + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def train_step(self, data, step):
        x, y, group = data
        with tf.GradientTape() as g:
            logits = self.classifier(x)
            entropy = utils.entropy_loss(logits, y)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('entropy-loss', entropy, step = step)
        
            