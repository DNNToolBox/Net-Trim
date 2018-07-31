"""
    Naive implementation of the Net-Trim for convolutional layer
    It converts the convolution into matrix multiplication and then runs the existing net-trim to solve it
"""
import tensorflow as tf
import numpy as np
import time
import NetTrimSolver_tf as nt_tf


def convert_image_to_patch(images, ksizes, strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding='VALID'):
    _graph = tf.Graph()

    with _graph.as_default():
        _X = tf.placeholder(tf.float64)

        # convert input images to patch, converting convolution into tensor multiplication
        _patch_X = tf.extract_image_patches(_X, ksizes=ksizes, strides=strides, rates=rates, padding=padding)

        _initializer = tf.global_variables_initializer()

    _sess = tf.Session(graph=_graph)
    _sess.run(_initializer)

    px = _sess.run(_patch_X, feed_dict={_X: images})

    h = np.ones(px.shape[:-1])
    px = np.concatenate((px, np.expand_dims(h, -1)), axis=3)
    px = np.reshape(px, newshape=(-1, px.shape[-1]))

    return px


def convolutional_nettrim_solver(X, Y, epsilon_gain, W, strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding='SAME'):
    tX = np.transpose(X, axes=(0, 2, 3, 1))
    tY = np.transpose(Y, axes=(0, 2, 3, 1))

    pX = convert_image_to_patch(tX, [1, W.shape[0], W.shape[1], 1], strides, rates, padding)
    pX = pX.transpose()
    pX = np.concatenate([pX, np.ones((1, pX.shape[1]))])

    pY = np.reshape(tY, newshape=(-1, tY.shape[-1]))
    pY = pY.transpose()

    unroll_number = 20
    num_iterations = 100
    nt = nt_tf.NetTrimSolver(unroll_number=unroll_number)

    V = np.zeros(pY.shape)
    epsilon = epsilon_gain * np.linalg.norm(pY)

    W_nt = nt.run(pX, pY, V, epsilon, rho=100, num_iterations=num_iterations)

    b_nt = W_nt[-1, :]
    W_nt = W_nt[:-1, :]
    W_nt = np.reshape(W_nt, newshape=(W.shape[0], W.shape[1], X.shape[1], -1))

    return W_nt, b_nt


