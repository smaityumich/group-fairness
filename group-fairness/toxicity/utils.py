import tensorflow as tf

@tf.function
def entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, \
        labels = tf.cast(labels[:,1], dtype = tf.int32)))

@tf.function
def logit_to_probability(logits):
    prob, _ = tf.linalg.normalize(tf.nn.softmax(logits), ord = 1, axis = 1)
    return prob

@tf.function 
def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis = 1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(labels[:,1], dtype = tf.int64)), dtype = tf.float32))


def label_specific_entropy_loss(logits, label):
    probabilities, _ = tf.linalg.normalize(tf.nn.softmax(logits), ord = 1, axis = 1)
    return tf.reduce_mean(-tf.math.log(probabilities[:, label]))


def false_negative_rate(labels, predictions):
    # Returns false negative rate for given labels and predictions.
    if tf.reduce_sum(labels > 0) == 0:  # Any positives?
        return 0.0
    else:
        return tf.reduce_mean(predictions[labels > 0] <= 0)


def false_positive_rate(labels, predictions):
    # Returns false positive rate for given labels and predictions.
    if tf.reduce_sum(labels <= 0) == 0:  # Any negatives?
        return 0.0
    else:
        return tf.reduce_mean(predictions[labels <= 0] > 0)


def group_false_negative_rates(labels, predictions, groups):
    # Returns list of per-group false negative rates for given labels,
    # predictions and group membership matrix.
    fnrs = []
    for ii in range(groups.shape[1]):
        labels_ii = labels[groups[:, ii] == 1]
        if tf.reduce_sum(labels_ii > 0) > 0:  # Any positives?
            predictions_ii = predictions[groups[:, ii] == 1]
            fnr_ii = tf.reduce_mean(predictions_ii[labels_ii > 0] <= 0)
        else:
            fnr_ii = 0.0
        fnrs.append(fnr_ii)
    return fnrs


def group_false_positive_rates(labels, predictions, groups):
    # Returns list of per-group false positive rates for given labels,
    # predictions and group membership matrix.
    fprs = []
    for ii in range(groups.shape[1]):
        labels_ii = labels[groups[:, ii] == 1]
        if tf.reduce_sum(labels_ii <= 0) > 0:  # Any negatives?
            predictions_ii = predictions[groups[:, ii] == 1]
            fpr_ii = tf.reduce_mean(predictions_ii[labels_ii <= 0] > 0)
        else:
            fpr_ii = 0.0
        fprs.append(fpr_ii)
    return fprs