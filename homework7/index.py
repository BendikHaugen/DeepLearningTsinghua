
from torch import batch_norm
from CCVariationalAutoencoder import build_VAE, build_q_net
from gradientdecent import gradient_decent
from helpers import load_mnist
import tensorflow as tf

print(tf.version.VERSION)
## Load MNIST
x_train, t_train, x_valid, t_valid, x_test, t_test, x_dim = load_mnist()

# Variables
z_dim=40 
epochs=100
batch_size=64

x_input= tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
n_particles=tf.placeholder(tf.int32, shape=[], name="n_particles")
x=tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input), tf.int32)
n=tf.placeholder(tf.int32, shape=[], name="n")



model = build_VAE(x_dim, z_dim, n, n_particles)
variational = build_q_net(x, z_dim, n_particles)

gradient_decent(
    epochs, 
    x_train,
    x_input,
    n_particles,
    n,
    model,
)
