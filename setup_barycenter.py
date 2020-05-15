import tensorflow as tf
import numpy as np
import datetime
import nn_graph
import utils



class GroupFairness():

    def __init__(self, batch_size = 500, epoch = 1000, learning_rate = 1e-4, wasserstein_lr = 1e-5, \
                        adversarial_learning_rate = 1e-3, adversarial_epoch = 20, \
                        l2_regularizer = 0, wasserstein_regularizer = 1e-2, \
                            epsilon = 1e0, clip_grad = 40, seed = 1, convex = False):

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
        self.adversarial_learning_rate = adversarial_learning_rate
        self.adversarial_epoch = adversarial_epoch
        self.classifier_optimizer = tf.optimizers.Adam(learning_rate)
        self.wasserstein_optimizer = tf.optimizers.Adam(wasserstein_lr)
        self.seed = seed
        self.convex = convex
        tf.random.set_seed(self.seed)

    def set_graph(self, classifier_architecture = [10, 20, 2], classifier_input_shape = (40, ),\
         n_groups = 3, potential_architecture = [10, 1], potential_input_shape = (1,),\
              activation = tf.nn.relu):

        if not n_groups > 1:
            raise ValueError('Number of groups must be an integer greater than or equal to 2')
        
        # Building classifier graph
        self.classifier = nn_graph.NNGraph(architecture=classifier_architecture, activation=activation, \
            input_shape=classifier_input_shape, name='classifier')

        # Building potential graphs
        self.potentials = []
        for label in [0,1]:
            potential_label = []
            for group in range((n_groups-1)):
                potential_label.append(nn_graph.NNGraph(potential_architecture, activation= activation,\
                 convex = self.convex, input_shape=potential_input_shape, name = f'potential-label{label}-group-{group}'))
            self.potentials.append(potential_label)

    def last_potential(self, x, potentials = None):
        '''
        Implements the constraint that sum of the potentials must add up to zero.
        Reference:
        [1] Agueh & Carlier: Barycenters in the Wasserstien space;
            https://www.ceremade.dauphine.fr/~carlier/AC_bary_Aug11_10.pdf, equation (2.3)
        '''

        if potentials == None:
            potentials = self.potentials[0]

        return_potentital = -tf.math.accumulate_n([potential(x) for potential in potentials])
        if self.convex:
            return_potentital += tf.reshape(tf.norm(x, axis = 1), shape = [-1, 1])

        return return_potentital

        

    def create_batch_data(self, data_train, data_test):
        # Tensor slices for train data
        batch = tf.data.Dataset.from_tensor_slices(data_train)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_train_data = batch.take(self.epoch)

        # Tensor slices for test data
        batch = tf.data.Dataset.from_tensor_slices(data_test)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_test_data = batch.take(self.epoch)


    def adversarial_objective(self, f, x, weight = 1, num_steps = 20, adversarial_learning_rate = 1e-4):
        '''
        Performs dual-wise adversarial optimization in equation (2.3), 
        Agueh & Carlier: Barycenters in the Wasserstien space
        '''
        
        x_start = tf.identity(x)
        for _ in range(num_steps):
            with tf.GradientTape() as g:
                g.watch(x)
                perturb = x - x_start
                if self.convex:
                    weighted_norm = (weight/2) * tf.reshape(tf.norm(perturb, axis = 1)**2\
                         - tf.norm(x, axis = 1)**2, shape = [-1, 1])
                    objective = tf.reduce_mean(weighted_norm + f(x)) # f(x) is weight/2 * |x|^2 - potential(x)
            
                else:
                    weighted_norm = (weight/2) * tf.reshape(tf.norm(perturb, axis = 1)**2, shape = [-1, 1])
                    objective = tf.reduce_mean(weighted_norm - f(x)) # f(x) is potential(x)

            gradient = g.gradient(objective, x)
            x = x - adversarial_learning_rate * gradient
        
        perturb = x - x_start
        if self.convex:
            weighted_norm = (weight/2) * tf.reshape(tf.norm(perturb, axis = 1)**2\
                         - tf.norm(x, axis = 1)**2, shape = [-1, 1])
            objective = tf.reduce_mean(weighted_norm + f(x))
            
        else:
            weighted_norm = (weight/2) * tf.reshape(tf.norm(perturb, axis = 1)**2, shape = [-1, 1])
            objective = tf.reduce_mean(weighted_norm - f(x))

        return objective



    def barycenter_wasserstein(self, grouped_data, potentials, num_steps = 20,\
         adversarial_learning_rate = 1e-4):

        n_groups = len(grouped_data)
        if len(potentials) != (n_groups - 1):
            raise TypeError('potentials list length must be one less than grouped data list length')
        

        def last_group_potential(x):
            return self.last_potential(x, potentials=potentials)

        #potentials.append(last_group_potential)

        # Calculating weights
        counts = tf.cast([data.shape[0] for data in grouped_data], dtype = tf.float32)
        weights = counts/tf.reduce_sum(counts)

        group_objectives = []
        for x, potential, weight in zip(grouped_data[:-1], potentials, weights[:-1]):
            group_objectives.append(self.adversarial_objective(potential, x,\
                 weight = weight, num_steps = num_steps,\
                      adversarial_learning_rate = adversarial_learning_rate))

        group_objectives.append(self.adversarial_objective(last_group_potential, grouped_data[-1],\
                 weight = weights[-1], num_steps = num_steps,\
                      adversarial_learning_rate = adversarial_learning_rate))

        return tf.math.add_n(group_objectives)




    def create_tensorboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        expt_id = np.random.randint(1000000)
        parameter = f'time-{current_time}-expt_id-{expt_id}-lr-{self.learning_rate}-w_reg-{self.epsilon}-l2_reg-{self.l2_regularizer}'
        self.train_log_dir = 'logs/' + parameter + '/train'
        self.test_log_dir = 'logs/' + parameter + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        print('parameter:' + parameter)




    def train_step(self, data, groups, step):
        
        x, y, group = data
        with tf.GradientTape(persistent = True) as g:
            logits = self.classifier(x)
            #entropy = utils.entropy_loss(logits, y) # for non-reweighted loss
            entropy = utils.label_specific_entropy_loss(logits[y[:, 1]==0], 0)\
                 + utils.label_specific_entropy_loss(logits[y[:, 1] == 1], 1)  # for reweighted loss
            
            accuracy = utils.accuracy(logits, y)

            # Entropy part of loss
            loss = tf.identity(entropy)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('entropy-loss', entropy, step = step)
                tf.summary.scalar('accuracy', accuracy, step = step)

            


            marginal_class_probabilities = tf.reduce_mean(y, axis = 0) # Calculates [P(Y = 0), P(Y = 1)]
            conditional_class_probabilities = utils.logit_to_probability(logits) # Calculates [P(Y = 0 | X = x), P(Y = 1 | X = x)]
            conditional_class_probabilities = tf.reshape(conditional_class_probabilities[:, 1], shape = [-1,1])

            # Wasserstein distance part of loss
            for label in [0, 1]:
                conditional_class_probabilities_given_label = conditional_class_probabilities[y[:, 1] == label]
                group_given_label = group[y[:, 1] == label]
                grouped_probabilities = [conditional_class_probabilities_given_label[tf.reduce_all(group_given_label\
                     == group_index, axis = 1)] for group_index in groups]
                barycenter_distance = self.barycenter_wasserstein(grouped_probabilities, self.potentials[label],\
                    adversarial_learning_rate=self.adversarial_learning_rate, num_steps=self.adversarial_epoch)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(f'barycenter-wasserstein-distance for y = {label}', barycenter_distance, step=step)
                loss = loss + self.wasserstein_regularizer * barycenter_distance * marginal_class_probabilities[label]

            # l2 norm part of loss
            if self.l2_regularizer > 0:
                norm = tf.cast(0, dtype = tf.dtypes.float32)
                trainable_vars_classifier = self.classifier.trainable_variables
                for v in trainable_vars_classifier:
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
                clipped_grad = [-tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
                self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, trainable_vars_dual_potential))
        del g



    def test_step(self, data, groups, step):

        x, y, group = data
        logits = self.classifier(x)
        #entropy = utils.entropy_loss(logits, y) # for non-reweighted loss
        
        entropy = utils.label_specific_entropy_loss(logits[y[:, 1]==0], 0)\
                 + utils.label_specific_entropy_loss(logits[y[:, 1] == 1], 1) # for reweighted loss

        accuracy = utils.accuracy(logits, y) 

        with self.test_summary_writer.as_default():
            tf.summary.scalar('entropy-loss', entropy, step=step)
            tf.summary.scalar('accuracy', accuracy, step = step)

        conditional_class_probabilities = utils.logit_to_probability(logits) 
        conditional_class_probabilities = tf.reshape(conditional_class_probabilities[:, 1], shape = [-1,1])

        # Wasserstein barycenter distance of test data
        for label in [0, 1]:
            conditional_class_probabilities_given_label = conditional_class_probabilities[y[:, 1] == label]
            group_given_label = group[y[:, 1] == label]
            grouped_probabilities = [conditional_class_probabilities_given_label[tf.reduce_all(group_given_label\
                 == group_index, axis = 1)] for group_index in groups]
            barycenter_distance = self.barycenter_wasserstein(grouped_probabilities, self.potentials[label],\
                adversarial_learning_rate=self.adversarial_learning_rate, num_steps=self.adversarial_epoch)
            with self.test_summary_writer.as_default():
                tf.summary.scalar(f'barycenter-wasserstein-distance for y = {label}', barycenter_distance, step=step)


    def gap_rms(self, predictions, y, group, target_feature):
        
        y = tf.cast(y[:, 1], dtype = tf.int32)
        rms = tf.cast(0, dtype = tf.float32)
        for label in [0, 1]:
            indices = (y == label)
            predictions_for_label = predictions[indices]
            group_for_label = group[indices]
            del indices
            predictions0 = predictions_for_label[group_for_label[:, target_feature] == 0]
            predictions1 = predictions_for_label[group_for_label[:, target_feature] == 1]
            predictions0, predictions1 = tf.cast(predictions0, dtype = tf.float32),\
                tf.cast(predictions1, dtype = tf.float32)
            rms += (tf.reduce_mean(predictions0) - tf.reduce_mean(predictions1))**2/2
        return rms
            


    def balanced_accuracy(self, y, predictions):

        
        #accuracy = utils.accuracy(logits, y).numpy()
        #predictions = tf.cast(tf.argmax(logits, axis = 1), dtype = tf.int32)
        y = tf.cast(y[:, 1], dtype = tf.int32)

        bal_accuracy = tf.cast(0, dtype = tf.float32)
        
        for label in [0, 1]:
            predictions_label = predictions[y == label]
            predictions_label = tf.cast(predictions_label, dtype = tf.float32)
            if label == 0:
                bal_accuracy = bal_accuracy + 1 - tf.reduce_mean(predictions_label)
            else:
                bal_accuracy = bal_accuracy + tf.reduce_mean(predictions_label)
        bal_accuracy = bal_accuracy/2

        return bal_accuracy

    def metrics(self, data, training_data = True, step = 0):
        
        # Training set
        x, y, group = data
        n_feature = group.shape[1]
        logits = self.classifier(x)
        probabilities = utils.logit_to_probability(logits)
        probabilities = tf.reshape(probabilities[:, 1], shape = [-1, 1])
        # Histograms for probabilities
        for label in [0, 1]:
            prob_for_label = probabilities[y[:, 1] == label]
            if training_data:
                with self.train_summary_writer.as_default():
                    tf.summary.histogram(f'conditional probability for y = {label}', prob_for_label, step=step)
            else:
                with self.test_summary_writer.as_default():
                    tf.summary.histogram(f'conditional probability for y = {label}', prob_for_label, step=step)


        accuracy = utils.accuracy(logits, y) # Accuracy
        predictions = tf.cast(tf.argmax(logits, axis = 1), dtype = tf.int32)
        bal_accuracy = self.balanced_accuracy(y, predictions)
        gap_rms = [self.gap_rms(predictions, y, group, target_feature) for target_feature in range(n_feature)]
        return accuracy, bal_accuracy, gap_rms

    
    def fit(self, data_train, data_test, groups):
        self.create_batch_data(data_train, data_test)
        self.create_tensorboard()
        for step, (batch_train_data, batch_test_data) in enumerate(zip(self.batch_train_data, self.batch_test_data)):
            self.train_step(batch_train_data, groups, step)
            self.test_step(batch_test_data, groups, step)
            
                



            
        
            