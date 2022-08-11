# -------------------------------------
# Project: DHN for biometrics
# Date: 2020.01.02
# Author: Tengfei Wu
# All Rights Reserved
# -------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os.path
import glob
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *
import time
import config
import numpy as np
from stn import spatial_transformer_network as stn_transformer

batch_size = config.batch_size
layer = 16
regularizer = tf.contrib.layers.l2_regularizer(0.0005)
#This version is the last fc code to 128 /64/256
# cancal added conv, code for 256


def encode(inputs, training=True, reuse=False, alpha=0.2):
    with tf.variable_scope('encode', reuse=reuse) as scope:
        with tf.variable_scope('stn', reuse=reuse) as scope:
            n_fc = 6
            if training:
                B, H, W, C = (batch_size, 128, 128, 1)
            else:
                B, H, W, C = (1, 128, 128, 1)

            # identity transform
            initial = np.array([[1., 0, 0], [0, 1., 0]])
            initial = initial.astype('float32').flatten()

            # input placeholder
            x = tf.placeholder(tf.float32, [B, H, W, C])

            # localization network
            W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
            b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
            h_fc1 = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1

            # spatial transformer layer
            stn_s = stn_transformer(inputs, h_fc1)

    with tf.variable_scope('encode1', reuse=reuse) as scope:
        weight1 = tf.get_variable('weight1', [3, 3, 1, 8], tf.float32, tf.glorot_uniform_initializer())  # 11 is 0.7%  5 is 0.5% 8000 trains
        bias1 = tf.get_variable('bias1', [8], tf.float32, tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=stn_s, filter=weight1, strides=[1, 2, 2, 1], padding='SAME', name='deconv1')
        mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.maximum(alpha*net, net)             #64
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')

    with tf.variable_scope('encode2', reuse=reuse) as scope:
        weight1 = tf.get_variable('weight1', [5, 5, 8, 16], tf.float32, tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1', [16], tf.float32, tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 2, 2, 1], padding='SAME', name='deconv1')
        mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.maximum(alpha*net, net)             #32
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')


    with tf.variable_scope('encode3', reuse=reuse) as scope:
        weight1 = tf.get_variable('weight1', [3, 3, 16, 32], tf.float32, tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1', [32], tf.float32, tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
        # mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        # net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.nn.bias_add(conv1, bias1)
        net = tf.maximum(alpha*net, net)
        # net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')


    with tf.variable_scope('encode4', reuse=reuse) as scope:
        weight1 = tf.get_variable('weight1', [3, 3, 32, 64], tf.float32, tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1', [64], tf.float32, tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
        # mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        # net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.nn.bias_add(conv1, bias1)
        net = tf.maximum(alpha*net, net)
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')             #3

    with tf.variable_scope('encode5', reuse=reuse) as scope:
        weight1 = tf.get_variable('weight1', [3, 3, 64, 128], tf.float32, tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1', [128], tf.float32, tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
        # mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        # net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.nn.bias_add(conv1, bias1)
        net = tf.maximum(alpha*net, net)
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')             #3


    return net

    # fc 训练时加入dropout
def encode6(inputs, training=True, reuse=False, alpha=0.2):

    with tf.variable_scope('encode6', reuse=reuse) as scope:

        code_shape = inputs.get_shape().as_list()
        weight1 = tf.get_variable('weight1', [code_shape[1], 2048], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1', [2048], tf.float32, initializer=tf.zeros_initializer())
        net = tf.matmul(inputs, weight1)
        #mean, variance = tf.nn.moments(net, [0, 1])
        #net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
        net = net + bias1
        net = tf.maximum(alpha*net, net)
        # if training: net = tf.nn.dropout(net,0.4)
        tf.add_to_collection('losses', regularizer(weight1))

    with tf.variable_scope('encode7', reuse=reuse) as scope:

        weight1 = tf.get_variable('weight1', [2048, 2048], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1', [2048], tf.float32, initializer=tf.zeros_initializer())
        net = tf.matmul(net, weight1)
        #mean, variance = tf.nn.moments(net, [0,1])
        #net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
        net = net + bias1
        net = tf.maximum(alpha*net, net)
        # if training: net = tf.nn.dropout(net,0.2)
        tf.add_to_collection('losses', regularizer(weight1))

    return net


def encode7(inputs, training=True, reuse=False, alpha=0.2):

    with tf.variable_scope('encode8', reuse=reuse) as scope:
        weight1 = tf.get_variable('weight1', [2048, 128], tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1', [128], tf.float32, initializer=tf.zeros_initializer())
        net = tf.matmul(inputs, weight1) + bias1
        net = tf.nn.tanh(net)

    return net


