#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:35:01 2020

@author: mdeb
"""
import tensorflow as tf
import numpy as np


'''
In this snippet we will try to calculate Wasserstein distance 
between two sets of 1-D vectors using Algorithm 1 from the paper
" LARGE-SCALE OPTIMAL TRANSPORT AND MAPPING ESTIMATION "
'''



tf.keras.backend.set_floatx('float32')

## NN for updating potential U

Evaluate_U = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape= (1,), input_shape=(1,)),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=192, activation='sigmoid'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1)
    
])


### NN for updating potential V
    
Evaluate_V = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(1,), input_shape=(1,)),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=192, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1)
    
])

optimizer = tf.keras.optimizers.Adam()





def cost_fn(x, y):
    '''
    Calculate the L2-cost function 
    '''
    x1 = tf.tile(tf.expand_dims(x, axis=1), [1, y.shape[0]])
    y1 = tf.tile(tf.expand_dims(y, axis=0), [x.shape[0], 1])
    return 0.5 * tf.square(tf.subtract(x1, y1))



def regularizer_matrix(u, v, cost_matrix, epsilon):
    #a = tf.math.l2_normalize(a)
    #b = tf.math.l2_normalize(b)
    
    M1 = tf.tile(tf.expand_dims(u, axis=1), [1, v.shape[0], 1])  # (na, nb, 2)
    M2 = tf.tile(tf.expand_dims(v, axis=0), [u.shape[0], 1, 1])
    cost_matrix = tf.dtypes.cast(tf.expand_dims(tf.convert_to_tensor(cost_matrix), axis = 2), tf.float32)  # (na, nb, 2)
    #M = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(M1, M2)),axis=2))  # (na, nb)
    M3 = tf.square(tf.math.maximum(tf.subtract(tf.add(M1, M2), cost_matrix),0))
    
    M4 = tf.math.negative(tf.math.divide(1.0, tf.math.multiply(4.0, 
                                                            epsilon)))
    M3 = tf.dtypes.cast(M3, tf.float32)
    M = tf.math.scalar_mul(M4, M3)
    
    return tf.squeeze(M,2)


def Dual_objective(u, v, cost_matrix, epsilon):
    '''
    This function calculates the dual objective function as mentioned in 
    the paper " LARGE-SCALE OPTIMAL TRANSPORT AND MAPPING ESTIMATION "
    '''
    u = tf.dtypes.cast(u, tf.float32)
    v = tf.dtypes.cast(v, tf.float32)
    regularization = tf.math.reduce_mean(regularizer_matrix(u,
                                    v, cost_matrix, epsilon))
    return tf.math.reduce_mean(u) + tf.math.reduce_mean(v) + regularization




def Reg_Wasserstein_distance(x, y, epsilon = 0.01, max_iter = 1000, tol = 1e-3):
    '''
    This function calculates the regularized-Wasserstein distance 
    between the empirical distribution of x and y
    '''
    cost_matrix = cost_fn(x, y)
    tol_level = 1e10
    iter = 0
    obj_old = 1e8
    while iter < max_iter and tol_level > tol:
        u = Evaluate_U(x)
        v = Evaluate_V(y)
        tol = -Dual_objective(u,v,cost_matrix,epsilon)
        with tf.GradientTape() as g:
            u = Evaluate_U(x)
            loss = -Dual_objective(u,v,cost_matrix,epsilon)
        trainable_variables_U = Evaluate_U.trainable_variables
        gradients_U = g.gradient(loss, trainable_variables_U)
        optimizer.apply_gradients(zip(gradients_U, trainable_variables_U))
        with tf.GradientTape() as g:
            v = Evaluate_V(y)
            loss = -Dual_objective(u,v,cost_matrix,epsilon)
        trainable_variables_V = Evaluate_V.trainable_variables
        gradients_V = g.gradient(loss, trainable_variables_V)
        optimizer.apply_gradients(zip(gradients_V, trainable_variables_V))
        iter += 1
        obj_now = loss
        tol_level = np.abs(obj_now - obj_old)
        obj_old = obj_now
    
    return iter, tol_level, obj_now




