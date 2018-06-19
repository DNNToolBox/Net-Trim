import time
import copy
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from CIFAR10ConvNet import BasicCIFAR10Model
from PrunedConvNet import PrunedConvNetModel
from CIFAR10DataBase import CIFAR10AugmentedDatabase
import NetTrimSolver_tf as nt_tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cifar10_folder = 'Z:/data/Database/CIFAR10/'


def load_network_parameters(file_name):
    if not os.path.exists(file_name):
        return None, None

    parameters = np.load(file_name, encoding='latin1')
    initial_weights = parameters['w']
    initial_biases = parameters['b']

    return initial_weights, initial_biases


def train_regulized_network(cifar10_db, l1_weight=None, l2_weight=None, keep_prob=1.0, file_name=None):
    if file_name is None:
        file_name = 'original_model.npz'

    print('=' * 60)
    print('l1 weight=', l1_weight, 'l2 weight=', l2_weight, 'keep prob.=', keep_prob)
    initial_weights, initial_biases = load_network_parameters(file_name)

    nn = BasicCIFAR10Model()
    nn.create_network(initial_weights, initial_biases)
    nn.add_regulizer(l1_weight, l2_weight)
    nn.create_optimizer(training_algorithm='Adam', learning_rate=0.001, decay_rate=0.98, decay_step=500)
    nn.create_initializer()

    nn.initialize()

    batch_size = 200
    iter_per_epoch = cifar10_db.training_number // batch_size
    max_epochs = 500
    test_images, test_labels = cifar10_db.get_test_data(one_hot=True)

    for epoch in range(max_epochs):
        t1 = time.time()
        for _ in range(iter_per_epoch):
            x, y = cifar10_db.get_training_batch(batch_size, one_hot=True, noise=0.001)
            nn.train(x, y, keep_prob)

        acc = nn.compute_accuracy(test_images, test_labels)
        t2 = time.time()
        print('{0:04d}: learning rate={1:5.4f}, accuracy={2:2.3f}, time={3:.3f} '.format(epoch, nn.learning_rate(), acc,
                                                                                         (t2 - t1) / iter_per_epoch))
        weights, biases = nn.get_weights()
        np.savez_compressed(file_name, w=weights, b=biases)

    result_labels = nn.get_output(test_images)
    cm = confusion_matrix(y_true=np.argmax(test_labels, axis=1), y_pred=np.argmax(result_labels, axis=1))
    for i in range(10):
        class_name = "({}) {}".format(i, cifar10_db.class_names[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({:04d})".format(i) for i in range(10)]
    print(''.join(class_numbers))

    weights, biases = nn.get_weights()
    np.savez_compressed(file_name, w=weights, b=biases)

    return weights, biases


def parallel_nettrim(epsilon_gain, original_weights, original_biases, signals, prune_layers):
    num_layers = len(original_weights)

    # pruning algorithm on fully connected layers
    unroll_number = 200
    num_iterations = 10
    nt = nt_tf.NetTrimSolver(unroll_number=unroll_number)

    pruned_weights = copy.deepcopy(original_weights)
    pruned_biases = copy.deepcopy(original_biases)

    for layer in range(num_layers):
        print(' Pruning layer ', layer)
        if not prune_layers[layer]:
            print('Convolutional layer: skipping.')
            continue

        num_samples = signals[layer].shape[0]
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


def fine_tuning(cifar10_db, weights, biases):
    weight_masks = [None] * len(weights)
    bias_masks = [None] * len(biases)

    for k in range(len(weights)):
        weight_masks[k] = np.ones(weights[k].shape)
        weight_masks[k][np.abs(weights[k]) < 1e-6] = 0

        bias_masks[k] = np.ones(biases[k].shape)
        bias_masks[k][np.abs(biases[k]) < 1e-6] = 0

    nn = PrunedConvNetModel()
    nn.create_network(weights, biases, weight_masks, bias_masks)
    nn.create_optimizer(training_algorithm='GD', learning_rate=0.005, decay_rate=0.98, decay_step=500)
    nn.create_initializer()

    nn.initialize()

    batch_size = 100
    iter_per_epoch = cifar10_db.training_number // batch_size
    max_epochs = 50
    test_images, test_labels = cifar10_db.get_test_data(one_hot=True)

    batch_size = 100
    acc = np.zeros(shape=max_epochs, dtype=np.float32)
    for epoch in range(max_epochs):
        t1 = time.time()
        for _ in range(iter_per_epoch):
            x, y = cifar10_db.get_training_batch(batch_size, one_hot=True, noise=0.0)
            nn.train(x, y)

        acc[epoch] = nn.compute_accuracy(test_images, test_labels)
        t2 = time.time()
        print('{0:04d}: learning rate={1:5.4f}, accuracy={2:2.3f}, time={3:.3f} '.format(epoch, nn.learning_rate(),
                                                                                         acc[epoch], (t2 - t1)))

    return acc


def compute_performance(weights, biases, training_images, test_images, test_labels):
    nn = BasicCIFAR10Model()
    nn.create_network(weights, biases)
    nn.create_initializer()

    nn.initialize()

    # compute the output of the network (prior to softmax) for the given training data
    Y = None
    batch_per_run = 10000
    index = 0
    while index < training_images.shape[0]:
        s = nn.get_fw_signals(training_images[index:(index + batch_per_run), :, :, :])
        if Y is None:
            Y = s[-2]
        else:
            Y = np.concatenate((Y, s[-2]), axis=0)

        index += batch_per_run

    # sparsity and accuracy
    acc = nn.compute_accuracy(test_images, test_labels)
    nnz = [np.count_nonzero(np.abs(w) > 1e-6) for w in weights]

    return acc, nnz, Y


def evaluate(epsilon_gains, l1_regulizer=0.0, l2_regulizer=0.004, keep_p=1.0, folder_name=''):
    """
    :type epsilon_gains: list
    :type l1_regulizer: float, list
    :type l2_regulizer: float, list
    :type keep_p: float
    :type folder_name: str
    """
    original_model = folder_name + 'original_model.npz'  # original network parameters

    print('Initializing database...')
    db = CIFAR10AugmentedDatabase(cifar10_folder, image_width=24, image_height=24)
    db.initialize_training_data()
    db.initialize_test_data()

    # get training samples used for net-trim evaluation
    num_samples = 5000  # db.training_number
    nt_training_x = None
    while num_samples > 0:
        db.reset_augmented_data()
        x, _ = db.get_training_batch(batch=num_samples, one_hot=True, noise=0.0)
        if nt_training_x is None:
            nt_training_x = x
        else:
            nt_training_x = np.concatenate((nt_training_x, x), axis=0)

        num_samples -= x.shape[0]

    test_image, test_label = db.get_test_data(one_hot=True)

    print('=' * 60)
    print('Loading parameters of the model...')
    original_weights, original_biases = load_network_parameters(original_model)
    if original_weights is None:
        print('The model is not trained. Training a new model...')
        original_weights, original_biases = train_regulized_network(db, l1_weight=l1_regulizer, l2_weight=l2_regulizer,
                                                                    keep_prob=keep_p, file_name=original_model)

    org_acc, org_nnz, org_Y = compute_performance(original_weights, original_biases, nt_training_x, test_image,
                                                  test_label)

    print('Computing signals in the neural network...')
    num_layers = len(original_weights)
    prune_layer = [False] * 2 + [True] * (num_layers - 1)

    nn = BasicCIFAR10Model()
    nn.create_network(original_weights, original_biases)
    nn.create_initializer()

    nn.initialize()

    signals = [None] * (num_layers + 1)
    batch_per_run = 10000
    index = 0
    while index < nt_training_x.shape[0]:
        s = nn.get_fw_signals(nt_training_x[index:(index + batch_per_run), :, :, :])
        for k in range(num_layers + 1):
            if prune_layer[k]:
                if signals[k] is None:
                    signals[k] = s[k]
                else:
                    signals[k] = np.concatenate((signals[k], s[k]), axis=0)

        index += batch_per_run

    pruning_all = []
    nt_disc_all = []
    nt_acc_all = []
    hptd_acc_all = []
    nt_ft_acc_all = []
    hptd_ft_acc_all = []

    for eps_gain in epsilon_gains:
        print('=' * 80)
        print('loading parameters of the pruned model for epsilon={} ...'.format(eps_gain))
        pruned_model = folder_name + 'parallel_{:.3f}.npz'.format(eps_gain)  # parameters of the pruned model
        nt_weights, nt_biases = load_network_parameters(pruned_model)
        if nt_weights is None:
            print('The model is not pruned. Running parallel nettrim with epsilon={}  ...'.format(eps_gain))
            nt_weights, nt_biases = parallel_nettrim(eps_gain, original_weights, original_biases, signals, prune_layer)
            np.savez_compressed(pruned_model, w=nt_weights, b=nt_biases)

        print('computing performance of the nettrim pruned model...')
        nt_acc, nt_nnz, nt_Y = compute_performance(nt_weights, nt_biases, nt_training_x, test_image, test_label)
        nt_disc = np.linalg.norm(nt_Y - org_Y) / np.linalg.norm(org_Y)  # discrepancy

        print('naive pruning of the parameters of the model...')
        print('number of non-zero weights per layer: ', nt_nnz)
        hptd_weights = naive_pruning(original_weights, nt_nnz)
        hptd_biases = original_biases
        hptd_acc, hptd_nnz, hptd_Y = compute_performance(hptd_weights, hptd_biases, nt_training_x, test_image,
                                                         test_label)

        # =====================================================
        # fine-tuning parameters
        print('=' * 60)
        print('fine-tuning parameters...')
        nt_ft_acc = fine_tuning(db, nt_weights, nt_biases)
        hptd_ft_acc = fine_tuning(db, hptd_weights, hptd_biases)

        total_pruning = 100.0 - np.sum(nt_nnz[2:]) * 100.0 / np.sum(org_nnz[2:])

        pruning_all += [total_pruning]
        nt_disc_all += [nt_disc]
        nt_acc_all += [nt_acc]
        hptd_acc_all += [hptd_acc]
        nt_ft_acc_all += [nt_ft_acc]
        hptd_ft_acc_all += [hptd_ft_acc]

    # displaying results
    print('=' * 60)
    str_eps = 'epsilon gain: ' + ', '.join('{:.2f}'.format(v) for v in epsilon_gains)
    str_org_acc = 'Initial model accuracy: {0:.2f}'.format(org_acc * 100.0)
    str_total_pruning = 'Total pruning: ' + ', '.join('{0:.2f}'.format(v) for v in pruning_all)
    str_nt_disc = 'NetTrim overall discrepancy: ' + ', '.join('{:.3f}'.format(v * 100.0) for v in nt_disc_all)
    str_nt_acc = 'NetTrim accuracy without fine-tuning: ' + ', '.join('{0:.2f}'.format(v * 100.0) for v in nt_acc_all)
    str_hptd_acc = 'HPTD accuracy without fine-tuning: ' + ', '.join('{0:.2f}'.format(v * 100.0) for v in hptd_acc_all)
    str_nt_ft_acc1 = 'NetTrim accuracy with fine-tuning (10 epochs): ' + ', '.join(
        '{0:.2f}'.format(v[10] * 100.0) for v in nt_ft_acc_all)
    str_hptd_ft_acc1 = 'HPTD accuracy with fine-tuning (10 epochs): ' + ', '.join(
        '{0:.2f}'.format(v[10] * 100.0) for v in hptd_ft_acc_all)
    str_nt_ft_acc2 = 'NetTrim accuracy with fine-tuning (20 epochs): ' + ', '.join(
        '{0:.2f}'.format(v[20] * 100.0) for v in nt_ft_acc_all)
    str_hptd_ft_acc2 = 'HPTD accuracy with fine-tuning (20 epochs): ' + ', '.join(
        '{0:.2f}'.format(v[20] * 100.0) for v in hptd_ft_acc_all)
    str_nt_ft_acc3 = 'NetTrim accuracy with fine-tuning (30 epochs): ' + ', '.join(
        '{0:.2f}'.format(v[30] * 100.0) for v in nt_ft_acc_all)
    str_hptd_ft_acc3 = 'HPTD accuracy with fine-tuning (30 epochs): ' + ', '.join(
        '{0:.2f}'.format(v[30] * 100.0) for v in hptd_ft_acc_all)

    print('l1 regulizer = ', l1_regulizer)
    print('l2 regulizer = ', l2_regulizer)
    print('keep prob. = ', keep_p)
    print('fine - tuning: SGD with 10/20/30 epochs of training')
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

    mat_file_name = folder_name + 'result.mat'
    sio.savemat(mat_file_name,
                {'l1': l1_regulizer, 'l2': l2_regulizer, 'keep_prob': keep_p, 'eps_gains': epsilon_gains,
                 'pruned': pruning_all, 'nt_disc': nt_disc_all, 'nt_acc': nt_acc_all, 'hptd_acc': hptd_acc_all,
                 'nt_ft_acc': np.stack(nt_ft_acc_all), 'hptd_ft_acc': np.stack(hptd_ft_acc_all)})


if __name__ == '__main__':
    base_folder = 'Z:/data/NetTrim/cifar10_convnet/'
    e = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # experiment 1
    result_folder = base_folder + 'exp1/'
    evaluate(e, 0.0, 0.0, 1.0, folder_name=result_folder)

    # experiment 2
    result_folder = base_folder + 'exp2/'
    evaluate(e, 0.0, [0.0, 0.0, 0.004, 0.004, 0.0], 1.0, folder_name=result_folder)

    # experiment 3
    result_folder = base_folder + 'exp3/'
    evaluate(e, 0.0, [0.0, 0.0, 0.004, 0.004, 0.004], 0.5, folder_name=result_folder)
