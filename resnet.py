"""This is a TensorFlow modified implementation of ResNet by He et all.

Paper: Deep Residual Learning for Image Recognition
(https://arxiv.org/abs/1512.03385)

Explanation on ResNet can be found in my blog post:
https://mohitjain.me/2018/06/13/resnet/
@author: Mohit Jain (contact: mohitjain1999(at)yahoo.com)

"""

import tensorflow as tf

from layers import conv_layer, max_pool, fc_layer, global_average
from residual import residual_block

class resnet(object):

	""" Implementation of ResNet Architecture """

	def __init__(self, x, n, num_classes):

		""" ResNet-n architecture
		{20:3, 32:5, 44:7, 56:9}
		"""

		if((n < 20) or ((n - 20) % 6 != 0)):
			print("ResNet DEPTH INVALID!\n")
			return

		self.NUM_CONV = int(((n - 20) / 6) + 3)
		self.X = x
		self.NUM_CLASSES = num_classes
		self.out = None

		self.create()

	def create(self):

		conv1 = conv_layer(self.X, 3, 3, 16, name = 'conv1')
		self.out = conv1

		for i in range(self.NUM_CONV):
			conv2 = residual_block(self.out, 16, name = 'conv2_1_{}'.format(i + 1))
			self.out = conv2

		pool2 = max_pool(self.out, name = 'pool2')
		self.out = pool2

		for i in range(self.NUM_CONV):
			conv3 = residual_block(self.out, 32, name = 'conv3_1_{}'.format(i + 1))
			self.out = conv3

		pool3 = max_pool(self.out, name = 'pool3')
		self.out = pool3

		for i in range(self.NUM_CONV):
			conv4 = residual_block(self.out, 64, name = 'conv4_1_{}'.format(i + 1))
			self.out = conv4

		global_pool = global_average(self.out, name = 'gap')
		self.out = global_pool

		flatten = tf.contrib.layers.flatten(self.out)
		fc5 = fc_layer(flatten, input_size = 64, output_size = self.NUM_CLASSES,
			relu = False, name = 'fc5')

		self.out = fc5
