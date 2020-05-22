import numpy as np
from adult_modified import preprocess_adult_data
import itertools


# Adult data processing
seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

all_train, all_test = dataset_orig_train.features, dataset_orig_test.features
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')

x_train = np.delete(all_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)
x_test = np.delete(all_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)

group_train = dataset_orig_train.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
group_test = dataset_orig_test.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]

groups = list(itertools.product([0,1], [0,1]))

groups_train = [x_train[np.all(group_train == g, axis = 1)] for g in groups]
groups_test = [x_test[np.all(group_test == g, axis = 1)] for g in groups]

data = x_test, y_test
grouped_data = group_test