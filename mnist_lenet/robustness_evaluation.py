import numpy as np
import glob
import scipy.io as sio
from BasicLenet import BasicLenetModel
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
    if 'type' in parameters.keys():
        layer_types = parameters['type'].astype('<U4').tolist()
    else:
        layer_types = ['conv', 'conv', 'fc', 'fc']

    return initial_weights, initial_biases, layer_types


def compute_performance(mnist, weights, biases, layer_types, noise_ratio=None, repeat=1):
    if noise_ratio is None:
        noise_ratio = [0.0]

    nn = BasicLenetModel()
    nn.create_network(weights, biases, layer_types)
    nn.create_initializer()

    nn.initialize()

    # compute power of input test images
    norm_x = np.linalg.norm(mnist.test.images, axis=1)

    acc = np.zeros(len(noise_ratio))

    for k, g in enumerate(noise_ratio):
        for _ in range(repeat):
            # add noise to the test images
            noise = np.random.normal(0, 1.0, mnist.test.images.shape)
            norm_n = np.linalg.norm(noise, axis=1)
            scale_n = g * norm_x / norm_n
            noise = noise * scale_n[:, np.newaxis]

            # compute accuracy
            acc[k] += nn.compute_accuracy(mnist.test.images + noise, mnist.test.labels)

        acc[k] /= repeat

    return acc


def main_func():
    mnist_folder = 'Z:/data/MNIST/'
    models_folder = 'Z:/data/NetTrim/mnist_lenet/dropout/exp{}/'
    noise_ratio = list(np.arange(0.0, 2.1, 0.1))

    np.set_printoptions(precision=2)
    mnist_db = input_data.read_data_sets(mnist_folder, one_hot=True)

    print('=' * 60)
    print("evaluating models in '{}'".format(models_folder))

    # get the list of all files in the folder
    model_files = glob.glob(models_folder + '*.npz')
    for model_name in model_files:
        print('loading parameters of the model "{}"'.format(model_name))
        weights, biases, layer_type = load_network_parameters(model_name)
        acc = compute_performance(mnist_db, weights, biases, layer_type, noise_ratio=noise_ratio, repeat=10)
        print('accuracy =', acc)

        mat_file_name = model_name[:-4] + '_robustness.mat'
        sio.savemat(mat_file_name, {'accuracy': acc, 'noise_ratio': noise_ratio})


if __name__ == '__main__':
    main_func()
