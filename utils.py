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