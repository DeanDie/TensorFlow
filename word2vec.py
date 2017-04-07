from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 10000
num_steps = 2000

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common[:vocabulary_size - 1])
    data = []
    dictionary = dict()
    for word, _ in range(count):
        dictionary[word] = len(dictionary)
    unk_count = 0
    for word in words:
        if word in dictionary.keys():
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reversed_dictionary = build_dataset(words, vocabulary_size)
del words
print("Build Finished!\nMost common words:", count[1:11])

index = 0

def generate_batches(batch_size, num_skips, skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    global index
    
    span = skip_window * 2 + 1
    buffer = collections.deque(maxlen=span)

    batches = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(span):
        buffer.append(data[index])
        index = (index + 1) % len(data)
    
    for i in rang(batch_size // num_skips):
        target =  skip_window
        target_to_avoid = [target]
        for j in random(num_skips):
            while(target in target_to_avoid):
                target = np.random.randint(0, span - 1)
            batches[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[target]
        buffer.append(data[index])
        index = (index + 1) % len(data)

    return batches, labels

batches, labels = generate_batches(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batches[i], dictionary[batches[i]], '->', labels[i], reversed_dictionary[labels[i]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window =100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    train_input = tf.placeholder(tf.int32, shape=[batch_size])
    train_label = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_input = tf.placeholder(tf.int32, shape=[valid_size])

    with tf.device('/cpu:0'):
        embeddings = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        embed = tf.nn.embedding_lookup(embeddings, train_input)
    
    nce_weight = tf.Variable(tf.truncated_normal(vocabulary_size, embedding_size), stddev = 1.0 / math.sqrt(embedding_size))
    nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights = nce_weight,
            biases = nce_bias,
            labels = train_label,
            inputs = train_input,
            num_sampled = num_sampled,
            num_classes = vocabulary_size
        )
    )
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)
    normalize_embedding = embeddings / norm
    valid_embedding = tf.nn.embedding_lookup(normalize_embedding, valid_input)
    similarity = tf.matmul(valid_embedding, normalize_embedding, transpose_b=True)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_input, batch_label = generate_batches(batch_size, num_skips, skip_window)
        _, cost = sess.run([optimizer, loss], feed_dict={train_input: batch_input, train_label: batch_label})
        average_loss += cost

        if step % 100 == 0:
            if step > 0:
                average_loss /= 100
                print('Average Loss at step', step, ":", average_loss, "(Every 500 step)")
                average_loss = 0
        if step % 500 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s" % valid_word
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = "%s %s " % (log_str, close_word)
                print(log_str)
