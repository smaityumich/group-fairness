import tensorflow as tf
import numpy as np
import datetime
import nn_graph
import utils
import toxicity_model


class GroupFairness():

    def __init__(self, batch_size = 500, epoch = 100, learning_rate = 1e-4, wasserstein_lr = 1e-5, \
                        l2_regularizer = 0, wasserstein_regularizer = 1e-2, \
                            epsilon = 1e0, clip_grad = 40, seed = 1,\
                                 wasserstein_regularization_type = 'L2', start_training = 50):

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
        self.wasserstein_regularization_type = wasserstein_regularization_type
        self.seed = seed
        self.start_training = start_training
        tf.random.set_seed(self.seed)

    def protected_groups(self, groups = None):
        if groups == None:
            self.group_names = ['sextuality', 'gender-identity', 'religion', 'race']
        else:
            self.group_names = groups

    def set_graph(self, embedding_matrix,\
         potential_architecture = [10, 1], potential_input_shape = (1,),\
              activation = tf.nn.relu):
        self.protected_groups()
        n_groups = len(self.group_names)

        if not n_groups > 1:
            raise ValueError('Number of groups must be an integer greater than or equal to 2')
        
        # Building classifier graph
        self.classifier = toxicity_model.toxicity_model(embedding_matrix)

        # Building potential graphs
        self.potentials = []
        for label in [0,1]:
            potential_label = []
            for group in self.group_names:
                potential_label.append(\
                    [nn_graph.NNGraph(potential_architecture, activation= activation,\
                input_shape=potential_input_shape, name = f'potential-label{label}-group-{group}-conditional'),\
                    nn_graph.NNGraph(potential_architecture, activation= activation,\
                input_shape=potential_input_shape, name = f'potential-label{label}-group-{group}-marginal'),\
                    ])
            self.potentials.append(potential_label)

    

        

    def create_batch_data(self, data_train, data_test):
        # Tensor slices for train data
        batch = tf.data.Dataset.from_tensor_slices(data_train)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_train_data = batch.take(self.epoch)

        # Tensor slices for test data
        batch = tf.data.Dataset.from_tensor_slices(data_test)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_test_data = batch.take(self.epoch)

        
    def create_sensitive_batch_data(self, sensitive_data_train, senssitive_data_test):
        ## Data stream for sensitive data

        # Tensor slices for train data
        batch = tf.data.Dataset.from_tensor_slices(sensitive_data_train)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_sensitive_train_data = batch.take(self.epoch)

        # Tensor slices for test data
        batch = tf.data.Dataset.from_tensor_slices(senssitive_data_test)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_sensitive_test_data = batch.take(self.epoch)

    def create_label_wise_batch_data(self, data_train, data_test):

        # Tensor slices for train data
        # creates label wise tensor slice
        x, y, group = data_train

        # Label 0
        index = y[:, 1] == 0
        data_train0 = x[index], y[index], group[index]
        batch = tf.data.Dataset.from_tensor_slices(data_train0)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_train_data0 = batch.take(self.epoch)

        # Label 1
        index = y[:, 1] == 1
        data_train1 = x[index], y[index], group[index]
        batch = tf.data.Dataset.from_tensor_slices(data_train1)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_train_data1 = batch.take(self.epoch)

        # Label wise tensor slices for test data
        x, y, group = data_test

        # Label 0
        # Label 0
        index = y[:, 1] == 0
        data_test0 = x[index], y[index], group[index]
        batch = tf.data.Dataset.from_tensor_slices(data_test0)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_test_data0 = batch.take(self.epoch)

        # Label 1
        index = y[:, 1] == 1
        data_test1 = x[index], y[index], group[index]
        batch = tf.data.Dataset.from_tensor_slices(data_test1)
        batch = batch.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.batch_test_data1 = batch.take(self.epoch)


    def wasserstein_distance(self, x1, x2, potential1 = None, potential2 = None, \
         reguralizer = 'entropy', update = True):
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
            potential1 = self.potentials[0][0][0]
        if potential2 == None:
            potential2 = self.potentials[0][0][1]
        
        if not update:
            u = potential1(x1)
            v = potential2(x2)
            distance = tf.cast(0, dtype = tf.float32)
            distance += tf.reduce_mean(u) + tf.reduce_mean(v) 
            if tf.math.is_nan(distance):
                distance = tf.cast(0, dtype = tf.float32)
            return distance
        else:
            with tf.GradientTape(persistent = True) as tape:
                u = potential1(x1)
                v = potential2(x2)
                distance = tf.reduce_mean(u) + tf.reduce_mean(v) 
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
                regularized_distance = distance + tf.reduce_mean(penalty)

            variables_u = potential1.trainable_variables
            gradient = tape.gradient(regularized_distance, variables_u)
            clipped_grad = [-tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
            self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, variables_u))

            variables_v = potential2.trainable_variables
            gradient = tape.gradient(regularized_distance, variables_v)
            clipped_grad = [-tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
            self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, variables_v))

            del tape

            if tf.math.is_nan(distance):
                distance = tf.cast(0, dtype = tf.float32)
                #print(f'Shape x1: '+str(x1.shape[0])+', x2: '+str(x2.shape[0]))
            
            return distance


    def marginal_OT(self, grouped_data, marginal_data, group_potentials, update = True):

        
        

        #OT_dist, reg_OT_dist = tf.cast(0, dtype = tf.float32), tf.cast(0, tf.float32)
        OT_dist = tf.cast(0, dtype = tf.float32)
        for x, potentials in zip(grouped_data, group_potentials):
            if True: #x.shape[0] != 0:
                wd = self.wasserstein_distance(x, marginal_data,\
             potential1=potentials[0], potential2=potentials[1],\
                  reguralizer=self.wasserstein_regularization_type, update = update)

                OT_dist += wd
            else:
                continue

        return OT_dist




    def create_tensorboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        expt_id = np.random.randint(1000000)
        parameter = f'seed-{self.seed}-time-{current_time}-expt_id-{expt_id}-lr-{self.learning_rate}-wlr-{self.wasserstein_lr}-epsilon-{self.epsilon}-w_reg-{self.wasserstein_regularizer}-l2_reg-{self.l2_regularizer}'
        self.train_log_dir = 'logs/' + parameter + '/train'
        self.test_log_dir = 'logs/' + parameter + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        #print('parameter:' + parameter)


    def update_potentials_only(self, data, step):
        x, y, group = data
        logits = self.classifier(x)
        conditional_class_probabilities = utils.logit_to_probability(logits) # Calculates [P(Y = 0 | X = x), P(Y = 1 | X = x)]
        conditional_class_probabilities = tf.reshape(conditional_class_probabilities[:, 1], shape = [-1,1])

        for label in [0, 1]:
            conditional_class_probabilities_given_label = conditional_class_probabilities[y[:, 1] == label]
            group_given_label = group[y[:, 1] == label]
            grouped_probabilities = [conditional_class_probabilities_given_label[group_given_label[:, i] == 1]\
                 for i, _ in enumerate(self.group_names)]
            OT = self.marginal_OT(grouped_probabilities,\
                     conditional_class_probabilities_given_label, self.potentials[label])
            with self.train_summary_writer.as_default():
                tf.summary.scalar(f'Marginal OT measure for y = {label}', OT, step=step)


    def train_step(self, data, step):
        
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
                grouped_probabilities = [conditional_class_probabilities_given_label[group_given_label[:, i] == 1]\
                 for i, _ in enumerate(self.group_names)]
                for _ in range(0):
                    _ = self.marginal_OT(grouped_probabilities,\
                     conditional_class_probabilities_given_label, self.potentials[label])
                OT = self.marginal_OT(grouped_probabilities,\
                     conditional_class_probabilities_given_label, self.potentials[label])
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(f'Marginal OT measure for y = {label}', OT, step=step)
                loss = loss + self.wasserstein_regularizer * OT * marginal_class_probabilities[label]

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

        #for potential_for_label in self.potentials:
        #    for potential_group in potential_for_label:
        #        for potential in potential_group:
        #            trainable_vars_dual_potential = potential.trainable_variables
        #            gradient = g.gradient(loss, trainable_vars_dual_potential)
        #            clipped_grad = [-tf.clip_by_norm(grad, self.clip_grad) for grad in gradient]
        #            self.wasserstein_optimizer.apply_gradients(zip(clipped_grad, trainable_vars_dual_potential))
        del g



    def test_step(self, data, step):

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
            grouped_probabilities = [conditional_class_probabilities_given_label[group_given_label[:, i] == 1]\
                 for i, _ in enumerate(self.group_names)]
            OT = self.marginal_OT(grouped_probabilities, conditional_class_probabilities_given_label,\
                 self.potentials[label], update=False)
            with self.test_summary_writer.as_default():
                tf.summary.scalar(f'Marginal OT measure for y = {label}', OT, step=step)


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
        return tf.sqrt(rms)
            


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
        n_feature = len(self.group_names)
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
        
        with self.train_summary_writer.as_default() if training_data else self.test_summary_writer.as_default():
            tf.summary.scalar('accuracy on full data', accuracy, step=step)
            tf.summary.scalar('balanced accuracy', bal_accuracy, step=step)
            for i, name in enumerate(self.group_names):
                tf.summary.scalar(f'gap rms for {name}', gap_rms[i], step=step)
        return accuracy, bal_accuracy, gap_rms

    
    def fit(self, data_train, data_test):
        self.create_batch_data(data_train, data_test)


        # Setting sensetive data stream
        text_train, labels_train, groups_train = data_train
        sensetive_indices_train = tf.reduce_sum(groups_train, axis= 1)>0
        sen_data_train = text_train[sensetive_indices_train], labels_train[sensetive_indices_train], \
            groups_train[sensetive_indices_train]

        text_test, labels_test, groups_test = data_test
        sensetive_indices_test = tf.reduce_sum(groups_test, axis= 1)>0
        sen_data_test = text_test[sensetive_indices_test], labels_test[sensetive_indices_test],\
         groups_test[sensetive_indices_test]

        self.create_sensitive_batch_data(sen_data_train, sen_data_test)

        self.create_tensorboard()
        # Creating part data test
        x, y, g = data_test
        
        sensetive_indices_test = tf.reduce_sum(g, axis= 1)>0
        xs, ys, gs = x[sensetive_indices_test], y[sensetive_indices_test], g[sensetive_indices_test]

        part_data_test = tf.concat([x[:800,:], xs[:800, :]], axis = 0), tf.concat([y[:800,:], ys[:800, :]], axis = 0),\
         tf.concat([g[:800,:], gs[:800, :]], axis = 0)



        for step, (batch_train_data, batch_test_data, batch_sensitive_train, batch_sensitive_test)\
             in enumerate(zip(self.batch_train_data, self.batch_test_data,\
                  self.batch_sensitive_train_data, self.batch_sensitive_test_data)):


            # Concating train data
            x, y, group = batch_train_data
            xs, ys, gs = batch_sensitive_train
            batch_train_data = tf.concat([x, xs], axis = 0), tf.concat([y, ys], axis = 0), tf.concat([group, gs], axis = 0)

            # Concating test data
            x, y, group = batch_test_data
            xs, ys, gs = batch_sensitive_test
            batch_test_data = tf.concat([x, xs], axis = 0), tf.concat([y, ys], axis = 0), tf.concat([group, gs], axis = 0)


            if step < self.start_training:
                self.update_potentials_only(batch_train_data, step)

            else:

                self.train_step(batch_train_data, step)
                self.test_step(batch_test_data, step)
                if step % 250 == 0:
                    
                    #_ = self.metrics(data_train, step=step)
                    accuracy, bal_accuracy, gap_rms = self.metrics(part_data_test, False, step)
                    print(f'Test accuracy for step {step}: {accuracy}\n')
                    for i, name in enumerate(self.group_names):
                        print(f'Test GAP RMS for {name} and step {step}: {gap_rms[i]}\n')
                    print(f'Test balanced accuracy for step {step}: {bal_accuracy}\n\n')
            print(f'Done step {step}')
        parameter = {'epsilon': self.epsilon, 'lr': self.learning_rate, 'wlr': self.wasserstein_lr,\
             'w_reg': self.wasserstein_regularizer}
        

        accuracy, bal_accuracy, gap_rms = self.metrics(data_test, False, step = self.epoch)
        print(str(parameter))
        parameter['test-acc'] = accuracy.numpy()
        for i, name in enumerate(self.group_names):
            parameter[f'test gap-rms-{name}'] = gap_rms[i].numpy()
        parameter['test bal-acc'] = bal_accuracy.numpy()
        print(f'Final test accuracy: {accuracy}')
        for i, name in enumerate(self.group_names):
            print(f'Final test GAP RMS for {name}: {gap_rms[i]}')
        print(f'Final test balanced accuracy: {bal_accuracy}\n\n')

        # False positive rate , and FNR
        x, y, groups = data_test
        groups = tf.cast(groups, dtype = tf.float32)
        labels = tf.cast(y[:, 1], dtype = tf.float32)
        predictions = tf.cast(tf.argmax(self.classifier(x), axis = 1), dtype = tf.float32)

        parameter['fpr'] = 1 - tf.reduce_mean(predictions[labels > 0]).numpy()
        parameter['fnr'] = tf.reduce_mean(predictions[labels <= 0]).numpy()
        #group_fprs = utils.group_false_positive_rates(labels, predictions, groups)
        #group_fnrs = utils.group_false_negative_rates(labels, predictions, groups)
        for i, name in enumerate(self.group_names):
            predictions_group = predictions[groups[:, i] == 1]
            labels_group = labels[groups[:, i] == 1]
            parameter[f'fpr-{name}'] = 1 - tf.reduce_mean(predictions_group[labels_group > 0]).numpy()
            parameter[f'fnr-{name}'] = tf.reduce_mean(predictions_group[labels_group <= 0]).numpy()

        with open('summary/toxicity1.out', 'a') as f:
            f.writelines(str(parameter) + '\n')
        #f.close()
        #print(parameter)
        # Saving model
        filename = f'saved-models/seed-{self.seed}-lr-{self.learning_rate}-wlr-{self.wasserstein_lr}-epsilon-{self.epsilon}-w_reg-{self.wasserstein_regularizer}-l2_reg-{self.l2_regularizer}'
        self.classifier.save(filename)

            
                



            
        
            
