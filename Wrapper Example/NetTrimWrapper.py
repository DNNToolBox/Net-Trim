import time
import numpy as np
import copy
from tensorflow.examples.tutorials.mnist import input_data
from BasicLenet import BasicLenetModel
from PrunedLenet import PrunedLenetModel
import NetTrimSolver_tf as nt_tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# list of parameters to tune:
# training batch size
batch_size = 200
# training Dropout probability
keep_prob = 0.75
# training max number of iterations
mx_iter = 2001

# Net-Trim parameters:
# number of loops inside GPU
unroll_number = 10
# number of loops outside the GPU
num_iterations = 30
# relative value of epsilon for Net-Trim
epsilon_gain = 0.15

# create neural network and train
nn = BasicLenetModel()
nn.create_network()
nn.create_optimizer(training_algorithm='Adam', learning_rate=0.001, decay_rate=0.98, decay_step=500)
nn.create_initializer()

nn.initialize()


for k in range(mx_iter):
    x, y = mnist.train.next_batch(batch_size)
    nn.train(x, y, keep_prob)

    if k % 500 == 0:
        acc = nn.compute_accuracy(mnist.validation.images, mnist.validation.labels)
        print('{0:2d}: learning rate={1:5.4f}, accuracy={2:2.3f} '.format(k // 500, nn.learning_rate(), acc))

org_acc = nn.compute_accuracy(mnist.test.images, mnist.test.labels)

#
# Net-Trim:
# change num_samples to a number, say 10000, if you want the Net-Trim retraining with only that many samples
num_samples = mnist.train.images.shape[0]
samples_x, _ = mnist.train.next_batch(num_samples)

orig_Weights, orig_Biases = nn.get_weights()
signals = nn.get_fw_signals(samples_x)
#
num_layers = len(orig_Weights)
#
# pruning algorithm on all layers


nt = nt_tf.NetTrimSolver(unroll_number=unroll_number)


layer_types = nn.get_layer_types()

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
    W_nt = nt.run(X, Y, V, epsilon, rho=1, num_iterations=num_iterations)
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

#
# Fine-Tuning Step on Top of Net-Trim
weight_masks = [None] * len(pruned_weights)
bias_masks = [None] * len(orig_Biases)

for k in range(len(orig_Weights)):
    weight_masks[k] = np.ones(pruned_weights[k].shape)
    weight_masks[k][np.abs(pruned_weights[k]) < 1e-6] = 0

    bias_masks[k] = np.ones(pruned_biases[k].shape)
    bias_masks[k][np.abs(pruned_biases[k]) < 1e-6] = 0

nn = PrunedLenetModel()
nn.create_network(pruned_weights, pruned_biases, layer_types, weight_masks, bias_masks)
nn.create_optimizer(training_algorithm='GD', learning_rate=0.01, decay_rate=0.98, decay_step=500)
nn.create_initializer()

nn.initialize()

nt_acc = nn.compute_accuracy(mnist.validation.images, mnist.validation.labels)
for k in range(mx_iter):
    x, y = mnist.train.next_batch(batch_size)
    nn.train(x, y)

ft_acc = nn.compute_accuracy(mnist.test.images, mnist.test.labels)
print("Accuracy of the original model: %.2f%%" % (100*org_acc))
print("Accuracy of the Net-Trim model: %.2f%%" % (100*nt_acc))
print("Accuracy of the Net-Trim fine-tuned model: %.2f%%" % (100*ft_acc))

org_nnz_str = 'original model: ' + ', '.join(['{}'.format(np.count_nonzero(abs(w)>1e-6)) for w in orig_Weights])
nt_nnz_str = 'Net-Trim pruned model: ' + ', '.join(['{}'.format(np.count_nonzero(abs(w)>1e-6)) for w in pruned_weights])

print('number of non-zeros per layer:')
print(org_nnz_str)
print(nt_nnz_str)

