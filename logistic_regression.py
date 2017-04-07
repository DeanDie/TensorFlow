import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
    
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

def model(x, W, b):
    return tf.matmul(x, W)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W = weight_variable([784, 10])

b = bias_variable([10])

y_ = model(X, W, b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
predict_op = tf.argmax(y_, 1)

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))