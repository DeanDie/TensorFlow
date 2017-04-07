import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

num_iter = 50
batch_size = 128

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01, name=name))

def model(x, W1, W2, W0, p_keep_input, p_keep_hidden):
    with tf.name_scope('layer1'):
        x = tf.nn.dropout(x, p_keep_input)
        h1 = tf.nn.relu(tf.matmul(x, W1))

    with tf.name_scope('layer2'):
        h1 = tf.nn.dropout(h1, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h1, W2))
    
    with tf.name_scope('layer3'):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, W0)

x = tf.placeholder('float', [None, 784], name='X')
y = tf.placeholder('float', [None, 10], name='Y')

W1 = init_weights([784, 625], 'W1')
W2 = init_weights([625, 625], 'W2')
W0 = init_weights([625, 10], 'W0')

tf.summary.histogram('w_1_summ', W1)
tf.summary.histogram('w_2_summ', W2)
tf.summary.histogram('w_0_summ', W0)

p_keep_input = tf.placeholder('float', name='p_keep_input')
p_keep_hidden = tf.placeholder('float', name='p_keep_hidden')

y_predict = model(x, W1, W2, W0, p_keep_input, p_keep_hidden)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=y))
    train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('Accuracy'):
    correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1)), 'float'))
    tf.summary.scalar('accuracy', correct)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs/nn_logs', sess.graph)
    merged = tf.summary.merge_all()

    tf.global_variables_initializer().run()

    for i in range(num_iter):
        for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size)):
            sess.run(train_op, feed_dict={x: train_x[start: end], y: train_y[start: end], p_keep_input: 0.8, p_keep_hidden: 0.6})
        summary, acc = sess.run([merged, correct], feed_dict={x: test_x, y: test_y, p_keep_input: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)

    print('--------------------Finished-------------------------')
    
    writer.close()