import six
from six.moves import cPickle as pickle
import numpy as np
import os
import random

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        if six.PY2:
            datadict = pickle.load(f)
        elif six.PY3:
            datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(root, num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = root
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test



def generate_batched_data(data, label, batch_size=10, shuffle=True):
    indices = list(range(data.shape[0]))
    if shuffle:
        random.shuffle(indices)
    data = data[indices, :, :, :]

    batched_data = []
    batched_label = []

    # num_batch = data.shape[0] / batch_size
    # while num_batch > 0:
    #     batch_mask = np.random.choice(data.shape[0], batch_size)
    #     batched_data.append(data[batch_mask])
    #     batched_label.append(label[batch_mask])
    #     num_batch -= 1

    start = 0
    while start < data.shape[0]:
        end = min(start+batch_size, data.shape[0])
        b_x = np.array(data[start:end])
        b_y = np.array(label[start:end])
        batched_data.append(b_x)
        batched_label.append(b_y)
        start = end
    return batched_data, batched_label





