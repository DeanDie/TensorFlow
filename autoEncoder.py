import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def encoder(x, weights, b):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encode_1']) + b['encode_1'])
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['encode_2']) + b['encode_2'])
    return layer_2

def decoder(x, weights, b):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decode_1']) + b['decode_1'])
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decode_2']) + b['decode_2'])
    return layer_2

def model(x, weights, b):
    encode_op = encoder(x, weights, b)
    decode_op = decoder(encode_op, weights, b)
    return decode_op

learning_rate = 0.01
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

x = tf.placeholder('float', [None, n_input])

weights = {
    'encode_1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encode_2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decode_1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decode_2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

b = {
    'encode_1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encode_2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decode_1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decode_2': tf.Variable(tf.random_normal([n_input]))
}

x_ = model(x, weights, b)

loss = tf.reduce_mean(tf.pow(x_ - x, 2))
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    total_batch = len(train_x) // batch_size
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, cost = sess.run([train_op, loss], feed_dict={x: batch_x})
        
        if epoch % display_step == 0:
            print("Epoch: %04d, cost: %.6f" % (epoch, cost))
    
    print("###################### OVER #######################")

    predict_x = sess.run(x_, feed_dict={x: test_x[:examples_to_show]})

    ######################################################################
    ######################################################################

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(test_x[i], (28, 28)))
        a[1][i].imshow(np.reshape(predict_x[i], (28, 28)))
    f.show()
    plt.show()
    plt.waitforbuttonpress()