# import tensorflow as tf
# import collections
# import numpy as np

# sentences = ["the quick brown fox jumped over the lazy dog",
#             "I love cats and dogs",
#             "we all love cats and dogs",
#             "cats and dogs are great",
#             "sung likes cats",
#             "she loves dogs",
#             "cats can be very independent",
#             "cats are great companions when they want to be",
#             "cats are playful",
#             "cats are natural hunters",
#             "It's raining cats and dogs",
#             "dogs and cats love sung"]

# # sentences to words and count
# words = " ".join(sentences).split()



# count = collections.Counter(words).most_common()

# dic = dict()
# for word, _ in count:
# 	dic[word] = len(dic)
# voc_size = len(dic)
# data = [dic[word] for word in words]
# print(data)

# x = tf.placeholder(tf.int32, shape=[len(words)])

# embedding = tf.Variable(tf.random_uniform([voc_size], -1.0, 1.0))
# embed = tf.nn.embedding_lookup(embedding, x)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(embed, feed_dict={x: data}))
# print(sorted(sess.run(embedding)))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np 

x = tf.placeholder('float')

num = 5.0

sess = tf.Session()
# sess.run(tf.global_variables_initializer())

print(sess.run(x, feed_dict={x: num}))