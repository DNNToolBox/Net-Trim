import tensorflow as tf
from .BasicFCnet import BasicFCSoftmaxModel


class RegularizedFCSodftmaxModel(BasicFCSoftmaxModel):
    # =========================================================================
    # define optimizer of the neural network
    def create_optimizer(self, training_algorithm='Adam', learning_rate=0.01, decay_rate=0.95, decay_step=100, rho=0.0):
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
            # training and gradients operators
            var_list = self._nn_weights + self._nn_biases

            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=rho, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, var_list)
            regularized_loss = self._loss + regularization_penalty
            self._trainOp = optimizer.minimize(regularized_loss, var_list=var_list, global_step=train_counter)

            gv = optimizer.compute_gradients(self._loss, var_list=var_list)
            self._gradients = [g for (g, v) in gv]
