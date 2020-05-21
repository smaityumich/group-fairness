import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import itertools
import wasserstein_fairness.basic_costs as costs
import wasserstein_fairness.combined_costs as combined_costs
import datetime



class unit_expt():

    def __init__(self, train_batch_size = 500, test_batch_size = 300, epoch = 10000, \
        alpha = 0.5, beta = 0.01, lambda_ = 0.1, learning_rate = 1e-3, seed = 1, clip = 40):
        super(unit_expt, self).__init__()


        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epoch = epoch
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.theta = np.random.random((40,))
        self.learning_rate = learning_rate
        groups = list(itertools.product([0,1], [0,1]))
        self.groups = [list(x) for x in groups]
        self.seed = seed
        self.clip = clip
        tf.random.set_seed(self.seed)

    def create_batch_data(self, data_train, data_test):
        # Tensor slices for train data
        batch = tf.data.Dataset.from_tensor_slices(data_train)
        batch = batch.repeat().shuffle(5000).batch(self.train_batch_size).prefetch(1)
        self.batch_train_data = batch.take(self.epoch)

        # Tensor slices for test data
        batch = tf.data.Dataset.from_tensor_slices(data_test)
        batch = batch.repeat().shuffle(5000).batch(self.test_batch_size).prefetch(1)
        self.batch_test_data = batch.take(self.epoch)


    def preprocess_data(self, data):
        x, y, group = data
        x, y, group = x.numpy(), y.numpy(), group.numpy()
        dataframe = pd.DataFrame(x), pd.DataFrame(y)
        groups_dataframe = [x[np.all(group == g, axis = 1)] for g in self.groups]
        return dataframe, groups_dataframe

    def create_tensorboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        expt_id = np.random.randint(1000000)
        parameter = f'seed-{self.seed}-time-{current_time}-expt_id-{expt_id}-lr-{self.learning_rate}-alpha-{self.alpha}-beta-{self.beta}-lambda-{self.lambda_}'
        self.train_log_dir = 'logs/' + parameter + '/train'
        self.test_log_dir = 'logs/' + parameter + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

    def accuracy(self, y_pred, y):
        return np.mean(y == y_pred)


    def train_step(self, data, step):
        x, y, group = data
        dataframe_train, groups_train = self.preprocess_data(data)
        
        logistic_grad, wasserstein_grad, logistic_objective, wasserstein_objective\
             = combined_costs.gradient_smoothed_logistic(dataframe_train, groups_train, theta = self.theta,\
                 lambda_ = self.lambda_, beta = self.beta, alpha= self.alpha)

        combined_grad = self.alpha * np.mean(logistic_grad, axis = 1) + (1-self.alpha) * wasserstein_grad
        clipped_grad = np.clip(combined_grad, -self.clip, self.clip)

        # Update step:
        self.theta -= self.learning_rate * clipped_grad

        x, y, group = x.numpy(), y.numpy(), group.numpy()
        y_pred = costs.predict(x, self.theta, 0.5)
        accuracy = np.mean(y_pred == y)
        
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Loss', logistic_objective, step=step)
            tf.summary.scalar('Wasserstein-distance', wasserstein_objective, step=step)
            tf.summary.scalar('Accuracy', accuracy, step=step)

    

    def gap_rms(self, predictions, y, group, target_feature):
        
        rms = 0
        for label in [0, 1]:
            indices = (y == label)
            predictions_for_label = predictions[indices]
            group_for_label = group[indices]
            del indices
            predictions0 = predictions_for_label[group_for_label[:, target_feature] == 0]
            predictions1 = predictions_for_label[group_for_label[:, target_feature] == 1]
            rms += (np.mean(predictions0) - np.mean(predictions1))**2/2
        return np.sqrt(rms)


    def balanced_accuracy(self, y, predictions):

        

        bal_accuracy = 0
        
        for label in [0, 1]:
            predictions_label = predictions[y == label]
            if label == 0:
                bal_accuracy = bal_accuracy + 1 - np.mean(predictions_label)
            else:
                bal_accuracy = bal_accuracy + np.mean(predictions_label)
        bal_accuracy = bal_accuracy/2

        return bal_accuracy


    def test_step(self, data, step):
        x, y, group = data
        x, y, group = x.numpy(), y.numpy(), group.numpy()
        y_pred = costs.predict(x, self.theta, 0.5)
        accuracy = np.mean(y_pred == y)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Accuracy', accuracy, step=step)


    def metrics(self, data, step, is_training = True):
        x, y, group = data
        x, y, group = x.numpy(), y.numpy(), group.numpy()
        y_pred = costs.predict(x, self.theta, 0.5)
        accuracy = np.mean(y == y_pred)
        bal_acc = self.balanced_accuracy(y, y_pred)
        gap_rms = [self.gap_rms(y_pred, y, group, i) for i in [0,1]]

        with self.train_summary_writer.as_default() if is_training else self.test_summary_writer.as_default():
            tf.summary.scalar('Accuracy full data', tf.cast(accuracy, dtype = tf.float32), step=step)
            tf.summary.scalar('balanced accuracy', tf.cast(bal_acc, dtype = tf.float32), step = step)
            for g in [0, 1]:
                tf.summary.scalar(f'gap rms for variable {g}', tf.cast(gap_rms[g], dtype = tf.float32), step = step)
        return accuracy, bal_acc, gap_rms

    
    def fit(self, data_train, data_test):
        names = ['gender', 'race']
        self.create_batch_data(data_train, data_test)
        self.create_tensorboard()
        for step, (batch_train_data, batch_test_data) in enumerate(zip(self.batch_train_data, self.batch_test_data)):
            self.train_step(batch_train_data, step)
            self.test_step(batch_test_data, step)
            if step % 250 == 0:
                _ = self.metrics(data_train, step= step)
                accuracy, bal_accuracy, gap_rms = self.metrics(data_test, step, False)
                print(f'Test accuracy for step {step}: {accuracy}\n')
                for i, name in enumerate(names):
                    print(f'Test GAP RMS for {name} and step {step}: {gap_rms[i]}\n')
                print(f'Test balanced accuracy for step {step}: {bal_accuracy}\n\n')
        accuracy, bal_acc, gap_rms = self.metrics(data_test, step, is_training=False)
        result_dict = {'alpha': self.alpha, 'beta': self.beta, 'lambda': self.lambda_, 'accuracy': accuracy,\
             'balanced accuracy': bal_acc}
        
        for i, name in enumerate(names):
            result_dict[f'gap-rms-{name}'] = gap_rms[i]
        with open('summary/adult-wfm3.out', 'a') as f:
            f.writelines(str(result_dict) + '\n')









    
