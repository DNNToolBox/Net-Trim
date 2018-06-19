import numpy as np
import scipy.linalg as sla


def net_trim_solver(X: np.ndarray, Y: np.ndarray, V, epsilon, rho, max_iteration=10000):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('Number of data sample for X and Y are not the same.')

    if V is None:
        # use simple sparse least-squares solution
        return sparse_least_squares_solver(X, Y, epsilon, rho, max_iteration)

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
    V = V / c0

    # Cholesky decomposition of C
    Lc = np.linalg.cholesky(C)
    Uc = Lc.transpose()

    Omega = np.where(Y > 1e-6)
    Omega_c = np.where(Y <= 1e-6)

    Y = Y[Omega]
    V = V[Omega_c]

    # initialization
    U1 = 0
    U2 = 0

    W1 = np.zeros(shape=(M, P))
    W3 = np.zeros(shape=(N, M))
    W2 = np.zeros(shape=(N, M))

    thr = M * N * 5e-9
    for cnt in range(max_iteration):
        Z = np.matmul(W3.transpose(), X) - U1

        W_prev = W3

        # a- updating W1[Omega]
        D = Z[Omega] - Y
        gain = epsilon / np.maximum(np.linalg.norm(D), epsilon)
        W1[Omega] = Y + gain * D

        # b- updating W1[Omega_c]
        W1[Omega_c] = Z[Omega_c] - np.maximum(Z[Omega_c] - V, 0)

        # c- updating W2
        D = W3 - U2
        W2 = np.sign(D) * np.maximum(np.abs(D) - 1 / rho, 0)

        # d- updating W3
        D = np.matmul(X, (W1 + U1).transpose()) + W2 + U2
        _1 = sla.solve_triangular(Lc, D, lower=True)
        W3 = sla.solve_triangular(Uc, _1, lower=False)

        # e- updating U1
        U1 = U1 + W1 - np.matmul(W3.transpose(), X)
        U2 = U2 + W2 - W3

        if np.linalg.norm(W3 - W_prev) < thr:
            break

        if cnt % 500 == 0:
            print('{0} : {1:3.6f}'.format(cnt, np.linalg.norm(W3 - W_prev)), flush=True)

    # ultimately, W2 and W3 would be the same
    return W2


def sparse_least_squares_solver(X: np.ndarray, Y: np.ndarray, epsilon, rho, max_iteration=10000):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('Number of data sample for X and Y are not the same.')

    # dimensions of signals
    N = X.shape[0]
    M = Y.shape[0]

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

    # initialization
    U1 = 0
    U2 = 0

    W3 = np.zeros(shape=(N, M))
    W2 = np.zeros(shape=(N, M))

    thr = M * N * 5e-9
    for cnt in range(max_iteration):
        Z = np.matmul(W3.transpose(), X) - U1

        W_prev = W3

        # a- updating W1
        D = Z - Y
        gain = epsilon / np.maximum(np.linalg.norm(D), epsilon)
        W1 = Y + gain * D

        # b- updating W2
        D = W3 - U2
        W2 = np.sign(D) * np.maximum(np.abs(D) - 1 / rho, 0)

        # c- updating W3
        D = np.matmul(X, (W1 + U1).transpose()) + W2 + U2
        _1 = sla.solve_triangular(Lc, D, lower=True)
        W3 = sla.solve_triangular(Uc, _1, lower=False)

        # d- updating U1, U2
        U1 = U1 + W1 - np.matmul(W3.transpose(), X)
        U2 = U2 + W2 - W3

        if np.linalg.norm(W3 - W_prev) < thr:
            break

        if cnt % 500 == 0:
            print('{0} : {1:3.6f}'.format(cnt, np.linalg.norm(W3 - W_prev)), flush=True)

    # ultimately, W2 and W3 would be the same
    return W2


def net_trim_solver_original(X: np.ndarray, Y: np.ndarray, V, epsilon, rho, max_iteration=10000):
    if X.shape[1] != Y.shape[1]:
        raise ValueError('Number of data sample for X and Y are not the same.')

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
    V = V / c0

    # Cholesky decomposition of C
    Lc = np.linalg.cholesky(C)
    Uc = Lc.transpose()

    Omega = np.where(Y > 1e-6)
    Omega_c = np.where(Y <= 1e-6)

    # initialization
    U1 = 0
    U2 = 0

    W1 = np.zeros(shape=(M, P))
    W3 = np.zeros(shape=(N, M))
    W2 = np.zeros(shape=(N, M))

    for cnt in range(max_iteration):
        Z = np.matmul(W3.transpose(), X) - U1

        W_prev = W3

        # a- updating W1[Omega]
        D = Z - Y
        norm_D = np.linalg.norm(D[Omega])
        if norm_D <= epsilon:
            W1[Omega] = Z[Omega]
        else:
            W1[Omega] = Y[Omega] + epsilon * D[Omega] / norm_D

        # b- updating W1[Omega_c]
        D = Z - V
        W1[Omega_c] = Z[Omega_c] - np.maximum(D[Omega_c], 0)

        # c- updating W2
        D = W3 - U2
        W2 = np.sign(D) * np.maximum(np.abs(D) - 1 / rho, 0)

        # d- updating W3
        D = np.matmul(X, (W1 + U1).transpose()) + W2 + U2
        _1 = sla.solve_triangular(Lc, D, lower=True)
        W3 = sla.solve_triangular(Uc, _1, lower=False)

        # e- updating U1
        U1 = U1 + W1 - np.matmul(W3.transpose(), X)
        U2 = U2 + W2 - W3

        if np.linalg.norm(W3 - W_prev) < 1e-4:
            break

    # ultimately, W2 and W3 would be the same
    return W2
