import tensorflow as tf 
import zhusuan as zs
import numpy as np
from time import *
import os
from skimage import io, img_as_ubyte

def gradient_decent(
    epochs, 
    x_train,
    x_test,
    x_input,
    n_particles,
    n,
    x,
    model,
    test_batch_size,
    variational,
    batch_size=10,
    lr=0.001,
    result_path = "results/vae"
    ):
    # Init variables
    infer_op = tf.train.AdamOptimizer(lr).minimize(cost)
    bn_gen = model.observe()
    lower_bound = tf.reduce_mean(lower_bound)
    cost = tf.reduce_mean(lower_bound.sgvb())
    is_log_likelihood = tf.reduce_mean(zs.is_loglikelihood(model, {"x": x}, proposal=variational, axis=0))


    with tf.Session() as S:
        S.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(x_train.shape[0]//test_batch_size):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = S.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            n_particles: 1,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            x_reshaped = tf.reshape(bn_gen["x_mean"], [-1, 28, 28, 1])
            
            if epoch % 10 == 0:
                time_test = -time()
                test_lbs, test_lls = [], []
                for t in range( x_test.shape[0] // test_batch_size):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = S.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1,
                                                  n: test_batch_size})
                    test_ll = S.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1000,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test lower bound = {}".format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))


            if epoch % 10 == 0:
                images = S.run(x_reshaped, feed_dict={n: 100, n_particles: 1})
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))

                if not os.path.exists(os.path.dirname(name)):
                    os.makedirs(os.path.dirname(name))
                n = x.shape[0]
                n_channels = x.shape[3]
                x = img_as_ubyte(x)
                r, c = (10, 10)
                h, w = x.shape[1:3]
                ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
                for i in range(r):
                    for j in range(c):
                        if i * c + j < n:
                            ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
                ret = ret.squeeze()
                io.imsave(name, ret)