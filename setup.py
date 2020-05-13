import tensorflow as tf
import numpy as np
import datetime
import nn_graph
import utils



class GroupFairness():

    def __init__(self, batch_size = 500, epoch = 1000, learning_rate = 1e-4, wasserstein_lr = 1e-5, \
                        l2_regularizer = 0, wasserstein_regularizer = 1e-2, epsilon = 1e0, clip_grad = 40, seed = 1):

        '''
        Class for enforcing group fairness in classifier

        parameters:
        batch_size: (int) size of individual batch data
        epoch: (int) total number of gradient descend iterations
        learning_rate: (float) learning rate for classifier
        waserstein_lr: (float) learning rate for dual potentials
        l2_regularizer: (float) regularization parameter for l2 penalty; activated only when positive
        wasserstein_regularizer: (float) regularization parameter for wasserstein distances
        epsilon: (float) regulrization parameter within wasserstein distances; see Seguy et. al.
        clip_grad: (float) parameter to clip gradient; uses tensorflow.clip_by_norm function
        seed: (int) 
        '''

        super(GroupFairness, self).__init__()
        self.batch_size = batch_size
        self.epoch = epoch
        self.l2_regularizer = l2_regularizer
        self.clip_grad = clip_grad
        self.learning_rate = learning_rate
        self.wasserstein_regularizer = wasserstein_regularizer
        self.wasserstein_lr = wasserstein_lr
        self.epsilon = epsilon
        self.classifier_optimizer = tf.optimizers.Adam(learning_rate)
        self.wasserstein_optimizer = tf.optimizers.Adam(wasserstein_lr)
        self.seed = seed
        tf.random.set_seed(self.seed)

    def set_graph(self, classifier_architecture = [10, 20, 2], classifier_input_shape = (40, ),\
         potential_architecture = [10, 1], potential_input_shape = (2,), activation = tf.nn.relu):

        self.classifier = nn_graph.NNGraph(architecture=classifier_architecture, activation=activation, \
            input_shape=classifier_input_shape, name='classifier')
        potential_y0 = [nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label0-group0'), \
                nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label0-group1')]
        potential_y1 = [nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label1-group0'), \
                nn_graph.NNGraph(architecture=potential_architecture, activation=activation, \
            input_shape=potential_input_shape, name = 'potential-label1-group1')]
        self.potentials = [potential_y0, potential_y1]

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
         reguralizer = 'entropy'):
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
        if potential1 == None:
            potential1 = self.potentials[0][0]
        if potential2 == None:
            potential2 = self.potentials[0][1]
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
            penalty = -self.epsilon * tf.exp((1/self.epsilon) * L)
        elif reguralizer == 'L2':
            penalty = -(1/(4*self.epsilon))*(tf.nn.relu(L)**2)
        else:
            raise TypeError('Wrong entry in regularizer. Options: entropy and L2')
        distance = tf.reduce_mean(u) + tf.reduce_mean(v) + tf.reduce_mean(penalty)
        return distance


    def create_tensorboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        expt_id = np.random.randint(1000000)
        parameter = f'-expt_id-{expt_id}-lr-{self.learning_rate}-w_reg-{self.epsilon}-l2_reg-{self.l2_regularizer}'
        self.train_log_dir = 'logs/time-' + current_time + parameter + '/train'
        self.test_log_dir = 'logs/time-' + current_time + parameter + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

    def train_step(self, data, step):
        
        #tf.summary.trace_on(graph = True, profiler = True)
        x, y, group = data
        with tf.GradientTape(persistent = True) as g:
            logits = self.classifier(x)
            #with self.train_summary_writer.as_default():
            #    tf.summary.trace_export(name = 'classifier-trace', step = step, profiler_outdir = self.train_log_dir)
        
            entropy = utils.entropy_loss(logits, y)
            accuracy = utils.accuracy(logits, y)

            # Entropy part of loss
            loss = tf.identity(entropy)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('entropy-loss', entropy, step = step)
                tf.summary.scalar('accuracy', accuracy, step = step)

            


            marginal_class_probabilities = tf.reduce_mean(y, axis = 0) # Calculates [P(Y = 0), P(Y = 1)]
            conditional_class_probabilities = utils.logit_to_probability(logits) # Calculates [P(Y = 0 | X = x), P(Y = 1 | X = x)]

            # Wasserstein distance part of loss
            for label in [0, 1]:
                conditional_class_probabilities_label = conditional_class_probabilities[y[:, 1] == label]
                group_label = group[y[:, 1] == label]
                probabilities_group0 = conditional_class_probabilities_label[group_label == 0]
                probabilities_group1 = conditional_class_probabilities_label[group_label == 1]
                wd = self.wasserstein_distance(probabilities_group0, \
                    probabilities_group1, potential1=self.potentials[label][0], \
                        potential2=self.potentials[label][1])
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(f'wasserstein-distance for y = {label}', wd, step=step)
                loss = loss + self.wasserstein_regularizer * wd * marginal_class_probabilities[label]

            # l2 norm part of loss
            if self.l2_regularizer > 0:
                norm = tf.cast(0, dtype = tf.dtypes.float32)
                trainable_vars_classifier = self.classifier.trainable_variables
                for v in trainable_vars_classifier:
                    norm = norm + tf.norm(v)
                for potential_list in self.potentials:
                    for potential in potential_list:
                        trainable_vars_dual_potential = potential.trainable_variables
                        for v in trainable_vars_dual_potential:
                            norm = norm + tf.norm(v)
                loss = loss + self.l2_regularizer * norm

        

        # updating classifier

        trainable_vars_classifier = self.classifier.trainable_variables
        gradient = g.gradient(loss, trainable_vars_classifier)
        clipped_grad = [tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
        self.classifier_optimizer.apply_gradients(zip(clipped_grad, trainable_vars_classifier))

        # updating dual potentials 

        for potential_list in self.potentials:
            for potential in potential_list:
                trainable_vars_dual_potential = potential.trainable_variables
                gradient = g.gradient(loss, trainable_vars_dual_potential)
                clipped_grad = [tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
                self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, trainable_vars_dual_potential))
        del g


    def test_step(self, data, step):

        x, y, _ = data
        logits = self.classifier(x)
        entropy = utils.entropy_loss(logits, y)
        accuracy = utils.accuracy(logits, y)
        with self.test_summary_writer.as_default():
            tf.summary.scalar('entropy-loss', entropy, step=step)
            tf.summary.scalar('accuracy', accuracy, step = step)
    
    def fit(self, data_train, data_test):
        self.create_batch_data(data_train, data_test)
        self.create_tensorboard()
        for step, (batch_train_data, batch_test_data) in enumerate(zip(self.batch_train_data, self.batch_test_data)):
            self.train_step(batch_train_data, step)
            self.test_step(batch_test_data, step)
            if step % 200 == 0:
                print(f'Done step {step}\n')

            
        
            