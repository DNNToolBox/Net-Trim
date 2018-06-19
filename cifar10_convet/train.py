import time
import numpy as np
from sklearn.metrics import confusion_matrix
import CIFAR10ConvNet as cnn
import CIFAR10DataBase as cifar10_db
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cifar10_folder = 'Z:/data/Database/CIFAR10/'
trained_folder = 'Z:/data/TrainedModels/CIFAR10/'
params_file_name = trained_folder + 'CIFAR10_CNN.npz'


def load_network_parameters(file_name):
    if not os.path.exists(file_name):
        return None, None

    parameters = np.load(file_name, encoding='latin1')
    initial_weights = parameters['w']
    initial_biases = parameters['b']

    return initial_weights, initial_biases


def train_network(file_name, l1_weight=None, l2_weight=None, keep_prob=0.75):
    db = cifar10_db.CIFAR10AugmentedDatabase(cifar10_folder, image_width=24, image_height=24)
    db.initialize_training_data()
    db.initialize_test_data()

    batch_size = 200
    iter_per_epoch = db.training_number // batch_size
    max_epochs = 500

    initial_weights, initial_biases = load_network_parameters(file_name)

    nn = cnn.BasicCIFAR10Model()
    nn.create_network(initial_weights, initial_biases)
    nn.add_regulizer(l1_weight, l2_weight)
    nn.create_optimizer(training_algorithm='Adam', learning_rate=0.001, decay_rate=0.98, decay_step=iter_per_epoch)
    nn.create_initializer()

    nn.initialize()

    test_images, test_labels = db.get_test_data(one_hot=True)

    for epoch in range(max_epochs):
        t1 = time.time()
        for _ in range(iter_per_epoch):
            x, y = db.get_training_batch(batch_size, one_hot=True, noise=0.001)
            nn.train(x, y, keep_prob)

        acc = nn.compute_accuracy(test_images, test_labels)
        t2 = time.time()
        print('{0:04d}: learning rate={1:5.4f}, accuracy={2:2.3f}, time={3:.3f} '.format(epoch, nn.learning_rate(), acc,
                                                                                         (t2 - t1) / iter_per_epoch))
        w, b = nn.get_weights()
        np.savez_compressed(file_name, w=w, b=b)

    result_labels = nn.get_output(test_images)
    cm = confusion_matrix(y_true=np.argmax(test_labels, axis=1), y_pred=np.argmax(result_labels, axis=1))
    for i in range(10):
        class_name = "({}) {}".format(i, db.class_names[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print(''.join(class_numbers))

    w, b = nn.get_weights()
    np.savez_compressed(file_name, w=w, b=b)


def compute_accuracy(file_name):
    initial_weights, initial_biases = load_network_parameters(file_name)

    nn = cnn.BasicCIFAR10Model()
    nn.create_network(initial_weights, initial_biases)
    nn.create_optimizer(training_algorithm='Adam', learning_rate=0.005, decay_rate=0.98, decay_step=500)
    nn.create_initializer()

    nn.initialize()

    db = cifar10_db.CIFAR10AugmentedDatabase(cifar10_folder)
    db.initialize_test_data()

    nn = cnn.BasicCIFAR10Model()
    nn.create_network(initial_weights, initial_biases)
    nn.create_initializer()

    nn.initialize()

    test_images, test_labels = db.get_test_data(one_hot=True)
    print('accuracy of the trained model = {.1f}'.format(nn.compute_accuracy(test_images, test_labels) * 100.0))


def db_test():
    db = cifar10_db.CIFAR10AugmentedDatabase()
    db.initialize_training_data()
    db.initialize_test_data()

    num_samples = 55000  # cifar10_db.training_number
    nt_training_x = None
    while num_samples > 0:
        db.reset_augmented_data()
        x, _ = db.get_training_batch(batch=num_samples, one_hot=True, noise=0.0)
        if nt_training_x is None:
            nt_training_x = x
        else:
            nt_training_x = np.concatenate((nt_training_x, x), axis=0)

        num_samples -= db.training_number

    print(nt_training_x.shape)


if __name__ == '__main__':
    db_test()
    # l1_weights = 0.0
    # l2_weights = 0.0
    # train_network(params_file_name, l1_weights, l2_weights, keep_prob=1.0)
    # compute_accuracy(params_file_name)
