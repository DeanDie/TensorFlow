import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# def init_bias(shape):
#     return tf.Variable(tf.zeros(shape))

def model(x, W, W_h):
    return tf.matmul(tf.nn.sigmoid(tf.matmul(x, W)), W_h)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = init_weight([784, 625])
W_h = init_weight([625, 10])

y_ = model(x, W, W_h)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
predict_op = tf.argmax(y_, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x) + 1, 128)):
            sess.run(train_op, feed_dict={x: train_x, y: train_y})
        print(i, np.mean(np.argmax(test_y, 1) == sess.run(predict_op, feed_dict={x: test_x})))