from layers import FCLayer, ReLULayer, ConvLayer, MaxPoolingLayer, ReshapeLayer, DropoutLayer
from criterion import SoftmaxCrossEntropyLossLayer
from optimizer import SGD
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from network import Network
from solver import train, test
from plot import plot_loss_and_acc
from mnist_data_loader import read_data_sets

batch_size = 100
max_epoch = 8
init_std = 0.01
learning_rate = 0.0025
weight_decay = 0.010
disp_freq = 10
# subtract by the mean value
dataset = read_data_sets("MNIST_data", one_hot=True, validation_size=5000)
mean_val = dataset.train.images.mean()
dataset.train._images -= mean_val
dataset.test._images -= mean_val
dataset.validation._images -= mean_val

# reshape to Nx1x28x28
dataset.train._images = dataset.train._images.reshape(-1, 1, 28, 28)
dataset.test._images = dataset.test._images.reshape(-1, 1, 28, 28)
dataset.validation._images = dataset.validation._images.reshape(-1, 1, 28, 28)

print("Dataset size: Training {}, Validation {}, Test {}"\
    .format(dataset.train.num_examples, dataset.validation.num_examples,
            dataset.test.num_examples))

criterion = SoftmaxCrossEntropyLossLayer()
sgd = SGD(learning_rate, weight_decay)
dropout_rate = 0.4
dropout = Network()
dropout.add(ConvLayer(1, 16, 3, 1))
dropout.add(ReLULayer())
dropout.add(MaxPoolingLayer(2, 0))
dropout.add(ConvLayer(16, 32, 3, 1))
dropout.add(ReLULayer())
dropout.add(MaxPoolingLayer(2, 0))
dropout.add(ReshapeLayer((batch_size, 32, 7, 7), (batch_size, 1568)))
dropout.add(FCLayer(1568, 128))
dropout.add(ReLULayer())
dropout.add(DropoutLayer(dropout_rate))
dropout.add(FCLayer(128, 64))
dropout.add(ReLULayer())
dropout.add(DropoutLayer(dropout_rate))
dropout.add(FCLayer(64, 32))
dropout.add(ReLULayer())
dropout.add(DropoutLayer(dropout_rate))
dropout.add(FCLayer(32, 10))


dropout.is_training = True
convNet, train_loss, train_acc, val_loss, val_acc = \
    train(dropout, criterion, sgd, dataset.train, dataset.validation, max_epoch, batch_size, disp_freq)