"""
    This is the main function to prune convolutional layers which uses an operator form of conjugate
    gradient as detailed in: https://dnntoolbox.github.io/Net-Trim/OperativeCG.pdf. Also available in the Journal version.
"""
import tensorflow as tf
import numpy as np


class NetTrimSolver:
    def __init__(self, unroll_number=10, cg_iter=10, strides=(1, 1, 1, 1), padding='SAME', precision=32):
        self._strides = strides
        self._padding = padding

        self._precision = tf.float32 if precision == 32 else tf.float64

        # create the computational graph
        self._graph = None
        self._sess = None
        self._initializer = None

        self._graph = tf.Graph()

        with self._graph.as_default():
            # fixed inputs to the algorithm
            self._X = tf.placeholder(self._precision)
            self._Y = tf.placeholder(self._precision)
            self._V = tf.placeholder(self._precision)
            self._b = tf.placeholder(self._precision)
            self._Mask = tf.placeholder(self._precision)

            # optimization parameters
            self._rho = tf.placeholder(dtype=self._precision)  # NOTE: this is 1/rho
            self._epsilon = tf.placeholder(dtype=self._precision)

            # variables of the algorithm, used for ADMM optimization
            self._init_W3 = tf.placeholder(self._precision)
            self._init_U1 = tf.placeholder(self._precision)
            self._init_U2 = tf.placeholder(self._precision)
            self._ZeroW = tf.zeros_like(self._init_W3)

            #
            Y1 = tf.multiply(self._Mask, self._Y)
            V1 = tf.multiply(1 - self._Mask, self._V)

            # create the main computational graph
            W3 = self._init_W3
            U1 = self._init_U1
            U2 = self._init_U2

            W2 = W3
            W3_prev = W3
            W3X = tf.add(tf.nn.conv2d(self._X, W3, strides=self._strides, padding=self._padding), self._b)

            for _ in range(unroll_number):
                Z = tf.subtract(W3X, U1)
                W3_prev = W3

                # update W1
                Zp = tf.multiply(self._Mask, Z)
                Zm = Z - Zp
                dY = Zp - Y1  # dY = tf.multiply(self._Mask, tf.subtract(Z, self._Y))
                dV = Zm - V1  # dV = tf.multiply(1 - self._Mask, tf.subtract(Z, self._V))

                gain = tf.divide(self._epsilon, tf.maximum(tf.norm(dY), self._epsilon))
                W1_p = Y1 + gain * dY  # tf.multiply(self._Mask, dY)
                W1_m = Zm - tf.maximum(dV, 0)  # tf.multiply(1 - self._Mask, Z - tf.maximum(dV, 0))
                W1 = W1_p + W1_m

                # update W2
                D = W3 - U2
                _1 = tf.maximum(tf.abs(D) - self._rho, 0)
                W2 = tf.multiply(tf.sign(D), _1)

                # update W3 using conjugate gradient
                W1U1 = W1 + U1
                W2U2 = W2 + U2
                W3 = self.conjugate_gradient_solver(B=W1U1, C=W2U2, W=W3, num_iterations=cg_iter)

                # update U1, U2
                W3X = tf.add(tf.nn.conv2d(self._X, W3, strides=self._strides, padding=self._padding), self._b)
                U1 = W1U1 - W3X
                U2 = W2U2 - W3

            self._U1 = U1
            self._U2 = U2
            self._W2 = W2
            self._W3 = W3
            self._dW3 = tf.norm(tf.subtract(W3, W3_prev))

            self._initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=self._graph)  # ,config=tf.ConfigProto(log_device_placement=True))

    def conjugate_gradient_solver(self, B, C, W, num_iterations=10):
        ZeroW = tf.zeros_like(W)

        # computing R0 = A*(B) + C
        _1 = tf.add(tf.nn.conv2d(self._X, ZeroW, strides=self._strides, padding=self._padding), self._b)
        _loss = tf.nn.l2_loss(_1 - B)
        R = C - tf.gradients(_loss, ZeroW)[0]
        P = R
        normR_new = tf.nn.l2_loss(R)

        # iterations to solve for min |X*W-B|^2 + |W-C|^2
        W = 0
        for _ in range(num_iterations):
            normR = normR_new
            T = tf.nn.conv2d(self._X, P, strides=self._strides, padding=self._padding)
            alpha = normR / (tf.nn.l2_loss(T) + tf.nn.l2_loss(P) + 1e-12)
            W = W + alpha * P

            _1 = tf.nn.conv2d(self._X, P, strides=self._strides, padding=self._padding)
            _loss = tf.nn.l2_loss(_1)
            R = R - alpha * (P + tf.gradients(_loss, P)[0])

            normR_new = tf.nn.l2_loss(R)
            beta = normR_new / normR
            P = R + beta * P

        return W

    def run(self, X, Y, V, bias, f_shape, epsilon, rho, num_iterations=100, verbose=False):
        c0 = np.linalg.norm(X) / np.sqrt(X.shape[0])
        epsilon = epsilon / c0
        X = X / c0
        Y = Y / c0
        bias = bias / c0

        if V is not None:
            V = V / c0
            Mask = np.zeros(Y.shape)
            Mask[Y > 1e-6] = 1
        else:
            V = np.zeros(Y.shape)
            Mask = np.ones(Y.shape)

        # Initialization
        U1 = np.zeros(shape=Y.shape)
        U2 = np.zeros(shape=f_shape)
        W2 = np.zeros(shape=f_shape)
        W3 = np.zeros(shape=f_shape)
        thr = np.prod(f_shape) * 1e-8

        self._sess.run(self._initializer)
        cnt = 0
        dw = 0
        for cnt in range(num_iterations):
            in_dict = {self._X: X, self._Y: Y, self._V: V, self._b: bias, self._Mask: Mask, self._rho: 1.0 / rho,
                       self._epsilon: epsilon, self._init_W3: W3, self._init_U1: U1, self._init_U2: U2}
            dw, W2, W3, U1, U2 = self._sess.run([self._dW3, self._W2, self._W3, self._U1, self._U2], feed_dict=in_dict)

            if verbose:
                print(' Iteration {0:02d}, error = {1:7.5f}'.format(cnt, np.linalg.norm(dw)))

            if dw < thr:
                break

        print(' Iteration {0:02d}, error = {1:7.5f}'.format(cnt, dw))
        return W2
