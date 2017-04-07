import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(x, W1, W2, W3, W4, W0, p_keep_prob, p_keep_hidden):
    conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME'))
    h1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h1 = tf.nn.dropout(h1, p_keep_prob)

    conv2 = tf.nn.relu(tf.nn.conv2d(h1, W2, strides=[1, 1, 1, 1], padding='SAME'))
    h2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h2 = tf.nn.dropout(h2, p_keep_prob)

    conv3 = tf.nn.relu(tf.nn.conv2d(h2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    h3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h3 = tf.reshape(h3, [-1, W4.get_shape().as_list()[0]])
    h3 = tf.nn.dropout(h3, p_keep_prob)

    h4 = tf.nn.relu(tf.matmul(h3, W4))
    h4 = tf.nn.dropout(h4, p_keep_hidden)

    return tf.matmul(h4, W0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

train_x = train_x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

x = tf.placeholder('float', [None, 28, 28, 1])
y = tf.placeholder('float', [None, 10])

W1 = init_weight([3, 3, 1, 32])
W2 = init_weight([3, 3, 32, 64])
W3 = init_weight([3, 3, 64, 128])
W4 = init_weight([128 * 4 * 4, 625])
W0 = init_weight([625, 10])

p_keep_prob = tf.placeholder('float')
p_keep_hidden = tf. placeholder('float')

y_ = model(x, W1, W2, W3, W4, W0, p_keep_prob, p_keep_hidden)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
predict_op = tf.argmax(y_, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(100):
        train_batch = zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size))
        for start, end in train_batch:
            # print(np.shape(train_x[start: end]))
            sess.run(train_op, feed_dict={x: train_x[start: end], y: train_y[start: end], p_keep_prob: 0.8, p_keep_hidden: 0.5})
        
        # test_indices = np.arange(len(test_x))
        # np.random.shuffle(test_indices)
        # test_indices = test_indices[0: test_size]
        test_indices = np.arange(len(test_x)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        tmp = sess.run(predict_op, feed_dict={x: test_x[test_indices], p_keep_prob: 1.0, p_keep_hidden: 1.0})

        print(i, np.mean(np.argmax(test_y[test_indices], axis=1) == tmp))
