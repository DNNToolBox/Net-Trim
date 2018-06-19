"""
   Basic convolutional neural network for classification of data.
   The default network creates a Lenet-5 like neural network for classification
   of MNSIT data.
"""

import tensorflow as tf
import numpy as np
from CIFAR10ConvNet import BasicCIFAR10Model


class PrunedConvNetModel(BasicCIFAR10Model):
    def __init__(self):
        BasicCIFAR10Model.__init__(self)

    # =========================================================================
    # build the neural network
    def create_network(self, initial_weights=None, initial_biases=None, weight_masks=None, bias_masks=None):
        if weight_masks is None:
            BasicCIFAR10Model.create_network(initial_weights, initial_biases)
        else:
            self._create_masked_network(initial_weights, initial_biases, weight_masks, bias_masks)

    def _add_masked_fully_connected_layer(self, x, weight, bias, weight_mask, bias_mask, func=''):
        w = tf.Variable(weight.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        mask_w = tf.constant(weight_mask, dtype=tf.float32)
        mask_b = tf.constant(bias_mask, dtype=tf.float32)

        self._fw_signals += [x]
        self._nn_weights += [w]
        self._nn_biases += [b]

        w = tf.multiply(w, mask_w)
        b = tf.multiply(b, mask_b)
        output = tf.matmul(x, w) + b
        if func == 'relu':
            output = tf.nn.relu(output)
        elif func == 'softmax':
            self._logit = output
            output = tf.nn.softmax(output)

        return output

    # create a fully connected neural network with given initial parameters
    def _create_masked_network(self, initial_weights, initial_biases, weight_masks, bias_masks):
        self._fw_signals = []
        self._nn_weights = []
        self._nn_biases = []

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._input = tf.placeholder(tf.float32, shape=[None, self._image_size, self._image_size, 3])
            self._target = tf.placeholder(tf.float32, shape=[None, 10])
            self._keep_prob = tf.placeholder(tf.float32)

            x = self._input
            # convolutional layer 1
            x = self._add_convolution_layer(x, initial_weights[0], initial_biases[0])
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # convolutional layer 2
            x = self._add_convolution_layer(x, initial_weights[1], initial_biases[1])
            x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_dim = initial_weights[2].shape[0]
            x = tf.reshape(x, [-1, flat_dim])
            x = self._add_masked_fully_connected_layer(x, initial_weights[2], initial_biases[2], weight_masks[2],
                                                       bias_masks[2], func='relu')
            x = self._add_masked_fully_connected_layer(x, initial_weights[3], initial_biases[3], weight_masks[3],
                                                       bias_masks[3], func='relu')
            x = self._add_masked_fully_connected_layer(x, initial_weights[4], initial_biases[4], weight_masks[4],
                                                       bias_masks[4], func='softmax')

            self._output = x
            self._fw_signals += [self._logit]
            self._fw_signals += [self._output]

            # loss function
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._target, logits=self._logit))

            # accuracy of the model
            matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
            self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
