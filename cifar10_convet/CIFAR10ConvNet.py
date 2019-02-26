import numpy as np
import scipy.stats as st
import tensorflow as tf


class BasicCIFAR10Model:
    def __init__(self):
        self._image_size = 24

        self._graph = None
        self._sess = None
        self._initializer = None
        self._accuracy = None

        self._trainOp = None
        self._learning_rate = 0.01
        self._loss = None
        self._gradients = None

        # parameters of the neural network
        self._keep_prob = None
        self._input = None
        self._output = None
        self._logit = None
        self._target = None

        self._nn_weights = []
        self._nn_biases = []
        self._fw_signals = []

    # ========================================================================
    # build the neural network
    def create_network(self, initial_weights=None, initial_biases=None):
        if initial_weights is None:
            self._create_random_network()
        else:
            self._create_initialized_network(initial_weights, initial_biases)

    def _add_convolution_layer(self, x, weight, bias):
        # convolution, followed by relu activation function
        w = tf.Variable(weight.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        self._fw_signals += [x]
        self._nn_weights += [w]
        self._nn_biases += [b]

        cnv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(cnv, b))

        return output

    def _add_fully_connected_layer(self, x, weight, bias, func=''):
        w = tf.Variable(weight.astype(np.float32), dtype=tf.float32)
        b = tf.Variable(bias.astype(np.float32), dtype=tf.float32)

        self._fw_signals += [x]
        self._nn_weights += [w]
        self._nn_biases += [b]

        x = tf.nn.dropout(x, self._keep_prob)
        output = tf.matmul(x, w) + b
        if func == 'relu':
            output = tf.nn.relu(output)
        elif func == 'softmax':
            self._logit = output
            output = tf.nn.softmax(output)

        return output

    # create neural network with random initial parameters
    def _create_random_network(self):
        flat_dim = self._image_size * self._image_size * 64 // 4 // 4
        layer_shapes = [[5, 5, 3, 64], [5, 5, 64, 64], [flat_dim, 384], [384, 192], [192, 10]]
        num_layers = len(layer_shapes)

        init_std = [0.05, 0.05, 0.04, 0.04, 1 / 192.0]
        init_bias = [0.0, 0.1, 0.1, 0.1, 0.0]
        initial_weights = [0] * num_layers
        initial_biases = [0] * num_layers
        # create initial parameters for the network
        for n in range(num_layers):
            initial_weights[n] = st.truncnorm(-2, 2, loc=0, scale=init_std[n]).rvs(layer_shapes[n])
            initial_biases[n] = np.ones(layer_shapes[n][-1]) * init_bias[n]

        self._create_initialized_network(initial_weights, initial_biases)
        return initial_weights, initial_biases

    # create a fully connected neural network with given initial parameters
    def _create_initialized_network(self, initial_weights, initial_biases):
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

            # fully connected layers: 3, 4, 5
            flat_dim = initial_weights[2].shape[0]
            x = tf.reshape(x, [-1, flat_dim])
            x = self._add_fully_connected_layer(x, initial_weights[2], initial_biases[2], func='relu')
            x = self._add_fully_connected_layer(x, initial_weights[3], initial_biases[3], func='relu')
            x = self._add_fully_connected_layer(x, initial_weights[4], initial_biases[4], func='softmax')

            self._output = x
            self._fw_signals += [self._logit]
            self._fw_signals += [self._output]

            # loss function
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._target, logits=self._logit))

            # accuracy of the model
            matches = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._logit, 1))
            self._accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    # =========================================================================
    # add regulizer to the loss function
    def add_regulizer(self, l1_weight=None, l2_weight=None):
        num_parameters = len(self._nn_weights)
        if l1_weight is None:
            l1_weight = [0.0] * num_parameters
        if l2_weight is None:
            l2_weight = [0.0] * num_parameters

        # no regularization for the convolutional layers by default
        if type(l1_weight) is float:
            l1_weight = [0.0] * 2 + [l1_weight] * (num_parameters - 2)

        if type(l2_weight) is float:
            l2_weight = [0.0] * 2 + [l2_weight] * (num_parameters - 2)

        with self._graph.as_default():
            l1_loss = tf.add_n([(s * tf.norm(w, ord=1)) for (w, s) in zip(self._nn_weights, l1_weight)])
            l2_loss = tf.add_n([(s * tf.nn.l2_loss(w)) for (w, s) in zip(self._nn_weights, l2_weight)])

            self._loss += (l1_loss + l2_loss)

    # =========================================================================
    # define optimizer of the neural network
    def create_optimizer(self, training_algorithm='Adam', learning_rate=0.01, decay_rate=0.95, decay_step=100):
        with self._graph.as_default():
            # define the learning rate
            train_counter = tf.Variable(0, dtype=tf.float32)
            # decayed_learning_rate = learning_rate * decay_rate ^ (train_counter // decay_step)
            self._learning_rate = tf.train.exponential_decay(learning_rate, train_counter, decay_step,
                                                             decay_rate=decay_rate, staircase=True)

            # define the appropriate optimizer to use
            if (training_algorithm == 0) or (training_algorithm == 'GD'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 1) or (training_algorithm == 'RMSProp'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 2) or (training_algorithm == 'Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 3) or (training_algorithm == 'AdaGrad'):
                optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 4) or (training_algorithm == 'AdaDelta'):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            else:
                raise ValueError("Unknown training algorithm.")

            # =================================================================
            # training and initialization operators
            var_list = self._nn_weights + self._nn_biases
            self._trainOp = optimizer.minimize(self._loss, var_list=var_list, global_step=train_counter)

            gv = optimizer.compute_gradients(self._loss, var_list=var_list)
            self._gradients = [g for (g, v) in gv]

    # =========================================================================
    # create initializer and session to run the network
    def create_initializer(self):
        # initializer of the neural network
        with self._graph.as_default():
            self._initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=self._graph)

    # =========================================================================
    # initialize the computation graph
    def initialize(self):
        if self._initializer is not None:
            self._sess.run(self._initializer)
        else:
            raise ValueError('Initializer has not been set.')

    # =========================================================================
    # compute the accuracy of the NN using the given inputs
    def compute_accuracy(self, x, target):
        return self._sess.run(self._accuracy, feed_dict={self._input: x, self._target: target, self._keep_prob: 1.0})

    # =========================================================================
    # One iteration of the training algorithm with input data
    def train(self, x, y, keep_prob=1.0):
        if self._trainOp is not None:
            self._sess.run(self._trainOp, feed_dict={self._input: x, self._target: y, self._keep_prob: keep_prob})
        else:
            raise ValueError('Training algorithm has not been set.')

    # =========================================================================
    # compute forward signals
    def get_fw_signals(self, x):
        return self._sess.run(self._fw_signals, feed_dict={self._input: x, self._keep_prob: 1.0})

    def get_output(self, x):
        return self._sess.run(self._output, feed_dict={self._input: x, self._keep_prob: 1.0})

    def get_weights(self):
        return self._sess.run([self._nn_weights, self._nn_biases])

    def learning_rate(self):
        return self._sess.run(self._learning_rate)
