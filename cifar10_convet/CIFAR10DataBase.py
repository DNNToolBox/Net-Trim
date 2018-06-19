import numpy as np
import pickle

default_folder = 'Z:/data/Database/CIFAR10/'


class CIFAR10Database(object):
    def __init__(self, database_folder=None):
        self._database_folder = database_folder

        if self._database_folder is None:
            self._database_folder = default_folder

        self._index_pos = 0
        self._shuffled_index = [0]
        self._training_number = 0
        self._train_x = None
        self._train_label = None

        self._test_number = 0
        self._test_x = None
        self._test_label = None

        f = open(self._database_folder + 'batches.meta', 'rb')
        d = pickle.load(f)
        f.close()
        self._class_names = d['label_names']

    @property
    def class_names(self):
        return self._class_names

    @property
    def training_number(self):
        return self._training_number

    def initialize_training_data(self):
        self._training_number = 0
        self._train_x = None
        self._train_label = None
        for i in range(5):
            f = open(self._database_folder + '/data_batch_' + str(i + 1), 'rb')
            d = pickle.load(f, encoding='latin1')
            f.close()

            _im = d['data'].reshape([-1, 3072])
            _im = _im.astype(float) / 255
            _r = _im[:, :1024].reshape([-1, 32, 32])
            _g = _im[:, 1024:2048].reshape([-1, 32, 32])
            _b = _im[:, 2048:].reshape([-1, 32, 32])
            im = np.stack((_r, _g, _b), axis=3)

            if self._train_x is None:
                self._train_x = im
                self._train_label = d['labels']
            else:
                self._train_x = np.concatenate((self._train_x, im), axis=0)
                self._train_label = np.concatenate((self._train_label, d['labels']), axis=0)

        self._training_number = self._train_x.shape[0]
        self._shuffled_index = np.arange(0, self._training_number)
        np.random.shuffle(self._shuffled_index)
        return self._training_number

    def initialize_test_data(self):
        f = open(self._database_folder + '/test_batch', 'rb')
        d = pickle.load(f, encoding='latin1')
        f.close()

        _im = d['data'].reshape([-1, 3072])
        _im = _im.astype(float) / 255
        _r = _im[:, :1024].reshape([-1, 32, 32])
        _g = _im[:, 1024:2048].reshape([-1, 32, 32])
        _b = _im[:, 2048:].reshape([-1, 32, 32])

        self._test_x = np.stack((_r, _g, _b), axis=3)
        self._test_label = [int(l) for l in d['labels']]
        self._test_number = self._test_x.shape[0]

        return self._test_number

    def get_training_batch(self, batch=100, one_hot=True):
        if type(batch) is int:
            index = self._shuffled_index[self._index_pos:(self._index_pos + batch)]
            self._index_pos += batch
            if self._index_pos >= self._training_number:
                # update the shuffled index
                self._index_pos = 0
                self._shuffled_index = np.arange(0, self._training_number)
                np.random.shuffle(self._shuffled_index)
        elif (type(batch) is list) or (type(batch) is np.ndarray):
            index = batch
        else:
            raise ValueError('Invalid input for batch.')

        num_samples = len(index)
        train_images = self._train_x[index, :, :, :]
        if one_hot:
            train_labels = np.zeros((num_samples, 10))
            train_labels[np.arange(0, num_samples), self._train_label[index]] = 1
        else:
            train_labels = self._train_label[index]

        return train_images, train_labels

    def get_test_data(self, one_hot=True):
        if one_hot:
            test_labels = np.zeros((self._test_number, 10))
            test_labels[np.arange(0, self._test_number), self._test_label] = 1
        else:
            test_labels = self._test_label

        return self._test_x, test_labels


class CIFAR10AugmentedDatabase(CIFAR10Database):
    def __init__(self, database_folder=None, image_width=24, image_height=24):
        super(CIFAR10AugmentedDatabase, self).__init__(database_folder)
        self._image_width = image_width
        self._image_height = image_height
        self._augmented_x = None

    def initialize_training_data(self):
        super(CIFAR10AugmentedDatabase, self).initialize_training_data()
        self.reset_augmented_data()

    def reset_augmented_data(self):
        self._augmented_x = np.zeros(
            shape=(self._training_number, self._image_height, self._image_width, self._train_x.shape[3]),
            dtype=np.float32)

        for idx in range(self._training_number):
            # random crop
            crop_u = np.random.randint(0, 32 - self._image_height)
            crop_b = crop_u + self._image_height
            crop_l = np.random.randint(0, 32 - self._image_width)
            crop_r = crop_l + self._image_width
            # random flip
            if np.random.randint(0, 2) == 0:
                self._augmented_x[idx, :, :, :] = self._train_x[idx, crop_u:crop_b, crop_l:crop_r, :]
            else:
                self._augmented_x[idx, :, :, :] = self._train_x[idx, crop_u:crop_b,
                                                                np.arange(crop_r - 1, crop_l - 1, -1), :]

        # image standardization
        mean_img = np.mean(self._augmented_x, axis=(1, 2))
        thr = 1.0 / np.sqrt(self._image_height * self._image_width)
        std_img = np.maximum(thr, np.std(self._augmented_x, axis=(1, 2)))
        self._augmented_x = (self._augmented_x - mean_img[:, np.newaxis, np.newaxis, :]) / std_img[:, np.newaxis,
                                                                                           np.newaxis, :]
        # shuffle
        self._index_pos = 0
        self._shuffled_index = np.arange(0, self._training_number)
        np.random.shuffle(self._shuffled_index)

    def get_training_batch(self, batch=100, one_hot=True, noise=0.0):
        if type(batch) is int:
            index = self._shuffled_index[self._index_pos:(self._index_pos + batch)]
            self._index_pos += batch
            if self._index_pos >= self._training_number:
                self.reset_augmented_data()  # reset the augmented data and index
        elif (type(batch) is list) or (type(batch) is np.ndarray):
            index = batch
        else:
            raise ValueError('Invalid input for batch.')

        num_samples = len(index)
        if one_hot:
            train_labels = np.zeros((num_samples, 10))
            train_labels[np.arange(0, num_samples), self._train_label[index]] = 1
        else:
            train_labels = self._train_label[index]

        train_images = self._augmented_x[index, :, :, :]
        # add noise
        train_images += np.random.normal(0, 1.0, train_images.shape) * noise

        return train_images, train_labels

    def get_test_data(self, one_hot=True):
        if one_hot:
            test_labels = np.zeros((self._test_number, 10))
            test_labels[np.arange(0, self._test_number), self._test_label] = 1
        else:
            test_labels = self._test_label

        # crop the center of the image to make the images of the desired size
        margin_u = (self._test_x.shape[1] - self._image_height) // 2
        margin_d = margin_u + self._image_height
        margin_l = (self._test_x.shape[2] - self._image_width) // 2
        margin_r = margin_l + self._image_width
        test_images = self._test_x[:, margin_u:margin_d, margin_l:margin_r, :]

        # image standardization
        mean_img = np.mean(test_images, axis=(1, 2))
        thr = 1.0 / np.sqrt(self._image_height * self._image_width)
        std_img = np.maximum(thr, np.std(test_images, axis=(1, 2)))
        test_images = (test_images - mean_img[:, np.newaxis, np.newaxis, :]) / std_img[:, np.newaxis, np.newaxis, :]

        return test_images, test_labels
