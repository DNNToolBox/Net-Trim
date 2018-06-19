import time
import numpy as np
import copy
from BasicLenet import BasicLenetModel
from PrunedLenet import PrunedLenetModel
import NetTrimSolver_tf as nt_tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_network_parameters(file_name):
    if not os.path.exists(file_name):
        return None, None, None

    parameters = np.load(file_name, encoding='latin1')
    initial_weights = parameters['w']
    initial_biases = parameters['b']
    layer_types = parameters['type'].astype('<U4').tolist()

    return initial_weights, initial_biases, layer_types


def train_regulized_network(mnist, l1_weight=None, l2_weight=None, keep_prob=1.0, file_name=None):
    if file_name is None:
        file_name = result_folder + 'original_model.npz'

    print('=' * 60)
    print('l1 weight=', l1_weight, 'l2 weight=', l2_weight, 'keep prob.=', keep_prob)
    initial_weights, initial_biases, layer_types = load_network_parameters(file_name)

    nn = BasicLenetModel()
    nn.create_network(initial_weights, initial_biases, layer_types)
    nn.add_regulizer(l1_weight, l2_weight)
    nn.create_optimizer(training_algorithm='Adam', learning_rate=0.001, decay_rate=0.98, decay_step=500)
    nn.create_initializer()

    nn.initialize()

    batch_size = 100
    for k in range(8001):
        x, y = mnist.train.next_batch(batch_size)
        nn.train(x, y, keep_prob)

        if k % 500 == 0:
            acc = nn.compute_accuracy(mnist.validation.images, mnist.validation.labels)
            print('{0:2d}: learning rate={1:5.4f}, accuracy={2:2.3f} '.format(k // 500, nn.learning_rate(), acc))

    weights, biases = nn.get_weights()
    layer_types = nn.get_layer_types()
    nz = [np.count_nonzero(np.abs(w) > 1e-6) for w in weights]
    acc = nn.compute_accuracy(mnist.test.images, mnist.test.labels)

    s = ', '.join('{:.0f}'.format(v) for v in nz)
    s = ' accuracy={0:4.2f}'.format(acc * 100) + ', number of non-zero elements: ' + s
    print(s)

    np.savez_compressed(file_name, w=weights, b=biases, type=layer_types)

    return weights, biases, layer_types


def parallel_nettrim(mnist, epsilon_gain, original_weights, original_biases, layer_types):
    nn = BasicLenetModel()
    nn.create_network(original_weights, original_biases, layer_types)
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
    num_iterations = 10
    nt = nt_tf.NetTrimSolver(unroll_number=unroll_number)

    pruned_weights = copy.deepcopy(orig_Weights)
    pruned_biases = copy.deepcopy(orig_Biases)

    for layer in range(num_layers):
        print(' Pruning layer ', layer)
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
        epsilon = epsilon_gain * norm_Y

        start = time.time()
        W_nt = nt.run(X, Y, V, epsilon, rho=100, num_iterations=num_iterations)
        elapsed = time.time() - start

        print('Elapsed time: {0:5.3f}'.format(elapsed))
        Y_nt = np.matmul(W_nt.transpose(), X)
        if layer < num_layers - 1:
            Y_nt = np.maximum(Y_nt, 0)

        rec_error = np.linalg.norm(Y - Y_nt)
        nz_count = np.count_nonzero(W_nt > 1e-6)
        print('non-zeros= {0}, epsilon= {1:.3f}, rec. error= {2:.3f}'.format(nz_count, epsilon, rec_error))
        pruned_weights[layer] = W_nt[:-1, :]
        pruned_biases[layer] = W_nt[-1, :]

    return pruned_weights, pruned_biases


def naive_pruning(weights, nnz):
    # threshold the weights such that only nnz elements are non-zero
    W = copy.deepcopy(weights)
    for k in range(len(W)):
        # find threshold to keep only nnz[k] elements
        q = 100.0 * (W[k].size - nnz[k]) / W[k].size
        thr = np.percentile(np.abs(W[k]), q)
        W[k][np.abs(W[k]) < thr] = 0.0

    return W


def fine_tuning(mnist, weights, biases, layer_types):
    weight_masks = [None] * len(weights)
    bias_masks = [None] * len(biases)

    for k in range(len(weights)):
        weight_masks[k] = np.ones(weights[k].shape)
        weight_masks[k][np.abs(weights[k]) < 1e-6] = 0

        bias_masks[k] = np.ones(biases[k].shape)
        bias_masks[k][np.abs(biases[k]) < 1e-6] = 0

    nn = PrunedLenetModel()
    nn.create_network(weights, biases, layer_types, weight_masks, bias_masks)
    nn.create_optimizer(training_algorithm='GD', learning_rate=0.01, decay_rate=0.98, decay_step=500)
    nn.create_initializer()

    nn.initialize()
    batch_size = 100
    acc1 = acc2 = acc3 = 0
    for k in range(15001):
        x, y = mnist.train.next_batch(batch_size)
        nn.train(x, y)

        if k % 500 == 0:
            acc = nn.compute_accuracy(mnist.validation.images, mnist.validation.labels)
            print('{0:2d}: learning rate={1:5.4f}, accuracy={2:.3f} '.format(k // 500, nn.learning_rate(), acc))

        if k == 5000:
            acc1 = nn.compute_accuracy(mnist.test.images, mnist.test.labels)
        elif k == 10000:
            acc2 = nn.compute_accuracy(mnist.test.images, mnist.test.labels)
        elif k == 15000:
            acc3 = nn.compute_accuracy(mnist.test.images, mnist.test.labels)

    return acc1, acc2, acc3


def compute_performance(mnist, weights, biases, layer_types):
    nn = BasicLenetModel()
    nn.create_network(weights, biases, layer_types)
    nn.create_initializer()

    nn.initialize()
    signals = nn.get_fw_signals(mnist.train.images)

    # sparsity and accuracy
    acc = nn.compute_accuracy(mnist.test.images, mnist.test.labels)
    nnz = [np.count_nonzero(np.abs(w) > 1e-6) for w in weights]

    return acc, nnz, signals[-2]


if __name__ == '__main__':
    epsilon_gains = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    l1_regulizer = 0.0
    l2_regulizer = 0.004
    keep_p = 0.5

    mnist_folder = 'Z:/data/MNIST/'
    result_folder = 'Z:/data/NetTrim/mnist_lenet/exp1/'
    original_model = result_folder + 'original_model.npz'  # original network parameters

    mnist_db = input_data.read_data_sets(mnist_folder, one_hot=True)

    print('=' * 60)
    print('loading parameters of the model...')
    org_weights, org_biases, org_types = load_network_parameters(original_model)
    if org_weights is None:
        print('The model is not trained. Training a new model...')
        org_weights, org_biases, org_types = train_regulized_network(mnist_db, l1_weight=0.0, l2_weight=0.004,
                                                                     keep_prob=0.5, file_name=original_model)

    org_acc, org_nnz, org_Y = compute_performance(mnist_db, org_weights, org_biases, org_types)

    pruning_all = []
    nt_disc_all = []
    nt_acc_all = []
    hptd_acc_all = []
    nt_ft_acc1_all = []
    hptd_ft_acc1_all = []
    nt_ft_acc2_all = []
    hptd_ft_acc2_all = []
    nt_ft_acc3_all = []
    hptd_ft_acc3_all = []

    for eps_gain in epsilon_gains:
        pruned_model = result_folder + 'parallel_{:.3f}.npz'.format(eps_gain)  # parameters of the pruned model
        print('loading parameters of the pruned model...')
        nt_weights, nt_biases, _ = load_network_parameters(pruned_model)
        if nt_weights is None:
            print('The model is not pruned. Running parallel nettrim with epsilon={}  ...'.format(eps_gain))
            nt_weights, nt_biases = parallel_nettrim(mnist_db, eps_gain, org_weights, org_biases, org_types)
            np.savez_compressed(pruned_model, w=nt_weights, b=nt_biases, type=org_types)

        print('computing performance of the nettrim pruned model...')
        org_acc, org_nnz, org_Y = compute_performance(mnist_db, org_weights, org_biases, org_types)
        nt_acc, nt_nnz, nt_Y = compute_performance(mnist_db, nt_weights, nt_biases, org_types)
        nt_disc = np.linalg.norm(nt_Y - org_Y) / np.linalg.norm(org_Y)  # discrepancy

        print('naive pruning of the parameters of the model...')
        print('number of non-zero weights per layer: ', nt_nnz)
        hptd_weights = naive_pruning(org_weights, nt_nnz)
        hptd_biases = org_biases
        hptd_acc, hptd_nnz, hptd_Y = compute_performance(mnist_db, hptd_weights, hptd_biases, org_types)

        # =====================================================
        # fine-tuning parameters
        print('=' * 60)
        print('fine-tuning parameters...')
        nt_ft_acc1, nt_ft_acc2, nt_ft_acc3 = fine_tuning(mnist_db, nt_weights, nt_biases, org_types)
        hptd_ft_acc1, hptd_ft_acc2, hptd_ft_acc3 = fine_tuning(mnist_db, hptd_weights, hptd_biases, org_types)

        total_pruning = 100.0 - np.sum(nt_nnz[2:]) * 100.0 / np.sum(org_nnz[2:])

        pruning_all += [total_pruning]
        nt_disc_all += [nt_disc]
        nt_acc_all += [nt_acc]
        hptd_acc_all += [hptd_acc]
        nt_ft_acc1_all += [nt_ft_acc1]
        nt_ft_acc2_all += [nt_ft_acc2]
        nt_ft_acc3_all += [nt_ft_acc3]
        hptd_ft_acc1_all += [hptd_ft_acc1]
        hptd_ft_acc2_all += [hptd_ft_acc2]
        hptd_ft_acc3_all += [hptd_ft_acc3]

    # displaying results
    print('=' * 60)
    str_eps = 'epsilon gain: ' + ', '.join('{:.2f}'.format(v) for v in epsilon_gains)
    str_org_acc = 'Initial model accuracy: {0:.2f}'.format(org_acc * 100.0)
    str_total_pruning = 'Total pruning: ' + ', '.join('{0:.2f}'.format(v) for v in pruning_all)
    str_nt_disc = 'NetTrim overall discrepancy: ' + ', '.join('{:.3f}'.format(v * 100.0) for v in nt_disc_all)
    str_nt_acc = 'NetTrim accuracy without fine-tuning: ' + ', '.join('{0:.2f}'.format(v*100.0) for v in nt_acc_all)
    str_hptd_acc = 'HPTD accuracy without fine-tuning: ' + ', '.join('{0:.2f}'.format(v*100.0) for v in hptd_acc_all)
    str_nt_ft_acc1 = 'NetTrim accuracy with fine-tuning (5000 iterations): ' + ', '.join('{0:.2f}'.format(v*100.0) for v in nt_ft_acc1_all)
    str_hptd_ft_acc1 = 'HPTD accuracy with fine-tuning (5000 iterations): ' + ', '.join('{0:.2f}'.format(v*100.0) for v in hptd_ft_acc1_all)
    str_nt_ft_acc2 = 'NetTrim accuracy with fine-tuning (10000 iterations): ' + ', '.join('{0:.2f}'.format(v*100.0) for v in nt_ft_acc2_all)
    str_hptd_ft_acc2 = 'HPTD accuracy with fine-tuning (10000 iterations): ' + ', '.join('{0:.2f}'.format(v*100.0) for v in hptd_ft_acc2_all)
    str_nt_ft_acc3 = 'NetTrim accuracy with fine-tuning (15000 iterations): ' + ', '.join('{0:.2f}'.format(v*100.0) for v in nt_ft_acc3_all)
    str_hptd_ft_acc3 = 'HPTD accuracy with fine-tuning (15000 iterations): ' + ', '.join('{0:.2f}'.format(v*100.0) for v in hptd_ft_acc3_all)

    print('l1 regulizer = ', l1_regulizer)
    print('l2 regulizer = ', l2_regulizer)
    print('keep prob. = ', keep_p)
    print('fine - tuning: SGD with 5000/10000/15000 iterations (10/20/30 epochs)')
    print('=' * 60)
    print(str_eps)
    print(str_org_acc)
    print(str_total_pruning)
    print(str_nt_disc)
    print(str_nt_acc)
    print(str_hptd_acc)
    print(str_nt_ft_acc1)
    print(str_hptd_ft_acc1)
    print(str_nt_ft_acc2)
    print(str_hptd_ft_acc2)
    print(str_nt_ft_acc3)
    print(str_hptd_ft_acc3)
