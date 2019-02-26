import tensorflow as tf
import numpy as np

# Main Net-Trim solver using Tensor Flow
class NetTrimSolver:
    def __init__(self, unroll_number=10):
        self._graph = None
        self._sess = None
        self._initializer = None

        self._graph = tf.Graph()

        with self._graph.as_default():
            # fixed inputs to the algorithm
            self._X = tf.placeholder(tf.float64)
            self._Y = tf.placeholder(tf.float64)
            self._V = tf.placeholder(tf.float64)
            self._Mask = tf.placeholder(tf.float64)
            self._Lc = tf.placeholder(tf.float64)
            self._Uc = tf.placeholder(tf.float64)

            # optimization parameters
            self._rho = tf.placeholder(dtype=tf.float64)  # NOTE: this is 1/rho
            self._epsilon = tf.placeholder(dtype=tf.float64)

            # variables of the algorithm, used for ADMM optimization
            self._init_W3 = tf.placeholder(tf.float64)
            self._init_U1 = tf.placeholder(tf.float64)
            self._init_U2 = tf.placeholder(tf.float64)

            #
            Y1 = tf.multiply(self._Mask, self._Y)
            V1 = tf.multiply(1 - self._Mask, self._V)
            # create the main computational graph
            W3 = self._init_W3
            U1 = self._init_U1
            U2 = self._init_U2
            W2 = W3
            W3_prev = W3
            W3tX = tf.matmul(tf.transpose(W3), self._X)
            for _ in range(unroll_number):
                Z = tf.subtract(W3tX, U1)
                W3_prev = W3

                Zp = tf.multiply(self._Mask, Z)
                dY = Zp - Y1  # dY = tf.multiply(self._Mask, tf.subtract(Z, self._Y))
                dV = Z - Zp - V1  # dV = tf.multiply(1 - self._Mask, tf.subtract(Z, self._V))

                # update W1
                gain = tf.divide(self._epsilon, tf.maximum(tf.norm(dY), self._epsilon))

                W1_p = Y1 + gain * tf.multiply(self._Mask, dY)  # tf.multiply(self._Mask, self._Y + gain * dY)
                W1_m = Z - Zp - tf.maximum(dV, 0)  # tf.multiply(1 - self._Mask, Z - tf.maximum(dV, 0))
                W1 = W1_p + W1_m

                # update W2
                D = W3 - U2
                _1 = tf.maximum(tf.abs(D) - self._rho, 0)
                W2 = tf.multiply(tf.sign(D), _1)

                # update W3
                W2U2 = W2 + U2
                W1U1 = W1 + U1
                _1 = tf.matmul(self._X, tf.transpose(W1U1)) + W2U2
                _2 = tf.matrix_triangular_solve(self._Lc, _1, lower=True)
                W3 = tf.matrix_triangular_solve(self._Uc, _2, lower=False)

                W3tX = tf.matmul(tf.transpose(W3), self._X)
                # update U1, U2
                U1 = W1U1 - W3tX
                U2 = W2U2 - W3

            self._U1 = U1
            self._U2 = U2
            self._W2 = W2
            self._W3 = W3
            self._dW3 = tf.norm(tf.subtract(W3, W3_prev))

            self._initializer = tf.global_variables_initializer()

        self._sess = tf.Session(graph=self._graph)  # ,config=tf.ConfigProto(log_device_placement=True))

    def run(self, X, Y, V, epsilon, rho, num_iterations=100):
        # dimensions of signals
        N = X.shape[0]
        M = Y.shape[0]
        P = X.shape[1]

        C = np.matmul(X, X.transpose())
        # normalize the input matrices and parameters
        c0 = 0.3 * np.sqrt(C.max())
        C = C / c0 ** 2 + np.eye(C.shape[0])

        epsilon = epsilon / c0
        X = X / c0
        Y = Y / c0

        # Cholesky decomposition of C
        Lc = np.linalg.cholesky(C)
        Uc = Lc.transpose()

        if V is not None:
            V = V / c0
            Mask = np.zeros(Y.shape)
            Mask[Y > 1e-6] = 1
        else:
            V = np.zeros(Y.shape)
            Mask = np.ones(Y.shape)

        # Initialization
        U1 = np.zeros(shape=(M, P))
        U2 = np.zeros(shape=(N, M))
        W2 = np.zeros(shape=(N, M))
        W3 = np.zeros(shape=(N, M))
        thr = M * N * 1e-8

        self._sess.run(self._initializer)
        cnt = 0
        dw = 0
        for cnt in range(num_iterations):
            # in_dict = {self._X: X, self._Y: Y, self._V: V, self._Mask: Mask, self._Lc: Lc, #self._Uc: Uc,
            #            self._rho: 1.0 / rho, self._epsilon: epsilon, self._init_W3: W3, self._init_U1: U1,
            #            self._init_U2: U2}

            in_dict = {self._X: X, self._Y: Y, self._V: V, self._Mask: Mask, self._Lc: Lc, self._Uc: Uc,
                       self._rho: 1.0 / rho, self._epsilon: epsilon, self._init_W3: W3, self._init_U1: U1,
                       self._init_U2: U2}
            dw, W2, W3, U1, U2 = self._sess.run([self._dW3, self._W2, self._W3, self._U1, self._U2], feed_dict=in_dict)

            if dw < thr:
                break

        print(' Iteration {0:02d}, error = {1:7.5f}'.format(cnt, np.linalg.norm(dw)))
        return W2
