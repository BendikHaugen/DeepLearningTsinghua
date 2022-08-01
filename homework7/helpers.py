import zhusuan as zs
import os
import numpy as np
import gzip
import pickle
import urllib
import six


def load_mnist():
    # Examples directory
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    # Dataset directory
    data_dir = os.path.join(examples_dir, "data")
    data_path = os.path.join(data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    x_dim = x_train.shape[1]
    return  x_train, t_train, x_valid, t_valid, x_test, t_test, x_dim


def load_mnist_realval(path):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.request.urlretrieve(url, path,)

    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]

    n_y = t_train.max() + 1
    t_transform =  (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)

def load_binary_mnist_realval(path):
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval(path)

    t_train = (t_train == 1).astype(np.float32)
    t_valid = (t_valid == 1).astype(np.float32)
    t_test = (t_test == 1).astype(np.float32)

    return x_train, t_train, x_valid, t_valid, x_test, t_test
