import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
train_x = train_x.reshape(-1, 28, 28)
test_x = test_x.reshape(-1, 28, 28)

batch_size = 128
lstm_size = 28
input_vec_size = 28
time_step_size = 28
test_size = 256
train_iter = 50
learning_rate = 0.01
weights_decay = 0.9

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))
    
def model(x, W, b, lstm_size):
    x_T = tf.transpose(x, [1, 0, 2])
    x_R = tf.reshape(x_T, [-1, lstm_size])
    x_spilt = tf.split(x_R, time_step_size, 0)

    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)

    output, state = rnn.static_rnn(lstm, x_spilt, dtype=tf.float32)

    return tf.matmul(output[-1], W) + b, lstm.state_size

x = tf.placeholder("float", [None, 28, 28])
y = tf.placeholder('float', [None, 10])

W = init_weight([lstm_size, 10])
b = init_weight([10])

y_predict, state_size = model(x, W, b, lstm_size)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=y)
train_op = tf.train.RMSPropOptimizer(learning_rate, weights_decay).minimize(loss)
predict_op = tf.argmax(y_predict, 1)

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

with tf.Session(config=session_conf) as sess:
    tf.global_variables_initializer().run()
    for i in range(train_iter):
        for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size)):
            sess.run(train_op, feed_dict={x:train_x[start: end], y:train_y[start: end]})
        
        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[:test_size]

        print(i, np.mean(np.argmax(test_y[test_indices], 1) == sess.run(predict_op, feed_dict={x: test_x[test_indices]})))