import time
import copy
import numpy as np
from models.BasicLenet import BasicLenetModel
from models.RegularizedLenet import RegularizedLenetModel
import NetTrimSolver_tf as nt_tf
from tensorflow.examples.tutorials.mnist import input_data
import fnmatch
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

trained_folder = '../data/TrainedModels/'
trained_model = trained_folder + 'Lenet.npz'
result_folder = '../data/TrimmedModels/Lenet/'
mnist_folder = '../data/MNIST/'


def load_network_parameters(file_name):
    if not os.path.exists(file_name):
        return None, None, None

    parameters = np.load(file_name, encoding='latin1')
    initial_weights = parameters['w']
    initial_biases = parameters['b']
    layer_types = parameters['type']

    return initial_weights, initial_biases, layer_types


def train_network(mnist):
    initial_weights, initial_biases, layer_types = load_network_parameters(trained_model)

    nn = BasicLenetModel()
    nn.create_network(initial_weights, initial_biases, layer_types)
    nn.create_optimizer(training_algorithm='Adam', learning_rate=0.001, decay_rate=0.97, decay_step=500)
    nn.create_initializer()

    nn.initialize()

    batch_size = 200
    for k in range(10001):
        x, y = mnist.train.next_batch(batch_size)
        nn.train(x, y, 1.0)

        if k % 1000 == 0:
            acc = nn.compute_accuracy(mnist.validation.images, mnist.validation.labels)
            print('{0:3d}: learning rate={1:5.4f}, accuracy={2:2.3f} '.format(k, nn.learning_rate(), acc))

    print(nn.compute_accuracy(mnist.test.images, mnist.test.labels))

    w, b = nn.get_weights()
    layer_types = nn.get_layer_types()
    np.savez_compressed(trained_model, w=w, b=b, type=layer_types)


def train_regulized_network(mnist, rho=0.0, keep_prob=1.0):
    print('=' * 40)
    print('l1 scale={0:.4f}, drop-out prob.={1:.2f}'.format(rho, keep_prob))
    file_name = trained_folder + 'Lenet/Params_{0:.4f},{1:.2f}.npz'.format(rho, keep_prob)
    initial_weights, initial_biases, layer_types = load_network_parameters(file_name)

    nn = RegularizedLenetModel()
    nn.create_network(initial_weights, initial_biases, layer_types)
    nn.create_optimizer(training_algorithm='Adam', learning_rate=0.001, decay_rate=0.97, decay_step=500, rho=rho)
    nn.create_initializer()

    nn.initialize()

    batch_size = 200
    for k in range(10001):
        x, y = mnist.train.next_batch(batch_size)
        nn.train(x, y, keep_prob)

        if k % 1000 == 0:
            acc = nn.compute_accuracy(mnist.validation.images, mnist.validation.labels)
            print('{0:3d}: learning rate={1:5.4f}, accuracy={2:2.3f} '.format(k, nn.learning_rate(), acc))

    weights, biases = nn.get_weights()
    layer_types = nn.get_layer_types()
    nz = [np.count_nonzero(np.abs(w) > 1e-6) for w in weights]
    acc = nn.compute_accuracy(mnist.test.images, mnist.test.labels)

    s = ', '.join('{:.0f}'.format(v) for v in nz)
    s = ' accuracy={0:4.2f}'.format(acc * 100) + ', number of non-zero elements: ' + s
    print(s)

    np.savez_compressed(file_name, w=weights, b=biases, type=layer_types)


def prune_lenet_parallel(mnist, epsilon_gain, rho=None, keep_prob=None):
    if rho is None:
        file_name = trained_model
    else:
        file_name = trained_folder + 'Lenet/Params_{0:.4f},{1:.2f}.npz'.format(rho, keep_prob)

    initial_weights, initial_biases, layer_types = load_network_parameters(file_name)
    if initial_weights is None:
        print('The model is not trained.')
        return

    nn = BasicLenetModel()
    nn.create_network(initial_weights, initial_biases, layer_types)
    nn.create_optimizer(training_algorithm='GD', learning_rate=0.01, decay_rate=0.97, decay_step=100)
    nn.create_initializer()

    nn.initialize()

    # use all training samples
    num_samples = mnist.train.images.shape[0]
    samples_x, _ = mnist.train.next_batch(num_samples)

    orig_Weights, orig_Biases = nn.get_weights()
    layer_types = nn.get_layer_types()
    signals = nn.get_fw_signals(samples_x)

    num_layers = len(orig_Weights)

    # pruning algorithm on all layers
    unroll_number = 200
    num_iterations = 25
    nt = nt_tf.NetTrimSolver(unroll_number=unroll_number)
    for g in epsilon_gain:
        print('======================================================')
        print('epsilon gain = {}'.format(g))

        # use deepcopy() to have parameters of the convolutional layers, which we are not going to prune
        pruned_Weights = copy.deepcopy(orig_Weights)
        pruned_biases = copy.deepcopy(orig_Biases)

        for layer in range(num_layers):
            print('\n Pruning layer {}'.format(layer))
            if layer_types[layer] == 'conv':
                print('Convolutional layer: skipping.')
                continue

            X = np.concatenate([signals[layer].transpose(), np.ones((1, num_samples))])
            Y = signals[layer + 1].transpose()

            if layer < num_layers - 1:
                # ReLU layer, use net-trim
                V = np.zeros(Y.shape)
            else:
                # use sparse least-squares (for softmax, ignore the activation function)
                V = None

            norm_Y = np.linalg.norm(Y)
            epsilon = g * norm_Y

            start = time.time()
            W_nt = nt.run(X, Y, V, epsilon, rho=100, num_iterations=num_iterations)
            elapsed = time.time() - start

            print('Elapsed time: {0:5.3f}'.format(elapsed))
            Y_nt = np.matmul(W_nt.transpose(), X)
            if layer < num_layers - 1:
                Y_nt = np.maximum(Y_nt, 0)

            rec_error = np.linalg.norm(Y - Y_nt)
            nz_count = np.count_nonzero(W_nt > 1e-6)
            print('non-zero elements= {0}, epsilon= {1:7.3f}, reconstruction error= {2:7.3f}'.format(nz_count, epsilon,
                                                                                                     rec_error))
            pruned_Weights[layer] = W_nt[:-1, :]
            pruned_biases[layer] = W_nt[-1, :]

        if rho is None:
            file_name = result_folder + 'parallel_g{0:.3f}.npz'.format(g)
        else:
            file_name = result_folder + 'parallel({0:.4f},{1:.2f})_g{2:.3f}.npz'.format(rho, keep_prob, g)

        np.savez_compressed(file_name, w=pruned_Weights, b=pruned_biases, type=layer_types)


def prune_lenet_cascade(mnist, epsilon_gain, rho=None, keep_prob=None):
    if rho is None:
        file_name = trained_model
    else:
        file_name = trained_folder + 'Lenet/Params_{0:.4f},{1:.2f}.npz'.format(rho, keep_prob)

    initial_weights, initial_biases, layer_types = load_network_parameters(file_name)
    if initial_weights is None:
        raise ValueError('The model is not trained.')

    nn = BasicLenetModel()
    nn.create_network(initial_weights, initial_biases, layer_types)
    nn.create_optimizer(training_algorithm='GD', learning_rate=0.01, decay_rate=0.97, decay_step=100)
    nn.create_initializer()

    nn.initialize()

    # use all training samples
    num_samples = mnist.train.images.shape[0]
    samples_x, _ = mnist.train.next_batch(num_samples)

    orig_Weights, orig_Biases = nn.get_weights()
    layer_types = nn.get_layer_types()
    signals = nn.get_fw_signals(samples_x)

    num_layers = len(orig_Weights)

    # pruning algorithm on all layers
    unroll_number = 200
    num_iterations = 25
    nt = nt_tf.NetTrimSolver(unroll_number=unroll_number)

    gamma = np.sqrt(1.1)
    for g in epsilon_gain:
        print('======================================================')
        print('epsilon gain = {}'.format(g))

        # initialize Y_hat
        Y_hat = None

        # use deepcopy() to have parameters of the convolutional layers, which we are not going to prune
        pruned_Weights = copy.deepcopy(orig_Weights)
        pruned_biases = copy.deepcopy(orig_Biases)

        for layer in range(num_layers):
            print('\n Pruning layer {}'.format(layer))
            if layer_types[layer] == 'conv':
                print('Convolutional layer: skipping.')
                continue

            if Y_hat is None:
                # initialize Y_hat to use for the next layer
                Y_hat = signals[layer].transpose()

            # original output of the layer: the BasicFCnet is modified to add the logit (pre-softmax) to the signals
            Y_orig = signals[layer + 1].transpose()

            # update Y_hat
            Y_hat = np.concatenate([Y_hat, np.ones((1, num_samples))], axis=0)

            W = np.concatenate([orig_Weights[layer], orig_Biases[layer][np.newaxis, :]], axis=0)
            V = np.matmul(W.transpose(), Y_hat)

            if layer == 0:
                epsilon = g * np.linalg.norm(Y_orig)
                V = np.zeros(Y_orig.shape)
            elif layer == (num_layers - 1):
                epsilon = gamma * np.linalg.norm(V - Y_orig)
                V = None
            else:
                Omega = np.where(Y_orig > 1e-6)
                epsilon = gamma * np.linalg.norm(V[Omega] - Y_orig[Omega])

            start = time.time()
            W_hat = nt.run(X=Y_hat, Y=Y_orig, V=V, epsilon=epsilon, rho=100, num_iterations=num_iterations)
            elapsed = time.time() - start
            print('Elapsed time: {0:5.3f}'.format(elapsed))

            # compute Y_hat
            Y_hat = np.maximum(np.matmul(W_hat.transpose(), Y_hat), 0)
            if layer < num_layers - 1:
                Y_hat = np.maximum(Y_hat, 0)

            rec_error = np.linalg.norm(Y_orig - Y_hat)
            nz_count = np.count_nonzero(W_hat > 1e-6)
            print('non-zero elements= {0}, epsilon= {1:7.3f}, reconstruction error= {2:7.3f}'.format(nz_count, epsilon,
                                                                                                     rec_error))

            pruned_Weights[layer] = W_hat[:-1, :]
            pruned_biases[layer] = W_hat[-1, :]

        if rho is None:
            file_name = result_folder + 'cascade_g{0:.3f}.npz'.format(g)
        else:
            file_name = result_folder + 'cascade({0:.4f},{1:.2f})_g{2:.3f}.npz'.format(rho, keep_prob, g)

        np.savez_compressed(file_name, w=pruned_Weights, b=pruned_biases, type=layer_types)


def compute_accuracy(mnist, parameter_folder):
    for f in os.listdir(parameter_folder):
        if fnmatch.fnmatch(f, '*.npz'):
            file_name = os.path.join(parameter_folder, f)
            if os.path.isfile(file_name):
                print(file_name)
                # load the parameters of the neural network
                initial_weights, initial_biases, layer_types = load_network_parameters(file_name)
                if initial_weights is None:
                    print('invalid file.')
                    continue

                nn = BasicLenetModel()
                nn.create_network(initial_weights, initial_biases, layer_types)
                nn.create_optimizer(training_algorithm='Adam', learning_rate=0.01, decay_rate=0.97, decay_step=100)
                nn.create_initializer()

                nn.initialize()

                acc = nn.compute_accuracy(mnist.test.images, mnist.test.labels)
                nz = [np.count_nonzero(abs(w) > 1e-6) for w in initial_weights]

                # parameters of the network
                info = re.findall('\d.\d+', f)
                if len(info) == 3:
                    info_str = 'l1 weight = {}, drop-out prob. = {}, epsilon gain={}'.format(info[0], info[1], info[2])
                elif len(info) == 2:
                    info_str = 'l1 weight = {}, drop-out prob. = {}'.format(info[0], info[1])
                else:
                    info_str = 'epsilon gain = {}'.format(info[0])

                # sparsity and accuracy
                s = ', '.join('{:.0f}'.format(v) for v in nz)
                s = ' accuracy={0:4.2f}'.format(acc * 100) + ', number of non-zero elements: ' + s
                print(info_str)
                print(s)
                print('=' * 40)


if __name__ == '__main__':
    mnist_db = input_data.read_data_sets(mnist_folder, one_hot=True)
    # train_network(mnist_db)

    # for p in [1.0, 0.75, 0.5]:
    #     for r in [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]:
    #         train_regulized_network(mnist_db, rho=r, keep_prob=p)

    # params_folder = trained_folder + 'Lenet/'
    params_folder = result_folder
    compute_accuracy(mnist_db, params_folder)

    # eps_gain_parallel = [0.01, 0.03, 0.04]
    # for p in [1.0, 0.75, 0.5]:
    #     for r in [0.0, 0.0001, 0.0005, 0.001, 0.002]:
    #         print('+' * 60)
    #         print('rho={0:.4f}, drop-out prob.={1:.2f}'.format(r, p))
    #         prune_lenet_parallel(mnist_db, eps_gain_parallel, rho=r, keep_prob=p)
    #
    # eps_gain_cascade = [0.005, 0.01, 0.03, 0.07]
    # for p in [1.0, 0.75, 0.5]:
    #     for r in [0.0, 0.0001, 0.0005, 0.001, 0.002]:
    #         print('+' * 60)
    #         print('rho={0:.4f}, drop-out prob.={1:.2f}'.format(r, p))
    #         prune_lenet_cascade(mnist_db, eps_gain_cascade, rho=r, keep_prob=p)
