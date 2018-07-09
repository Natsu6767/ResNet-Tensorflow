"""This is a TensorFlow implementation of ResNet by He et all.

The architecture is based on the ResNet-n architecture
(where n = 20, 32, 44, 56) described in the paper to
perform analysis and tests on CIFAR-10 dataset.

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

		self.NUM_CONV = int(((n - 20) / 6) + 3) # For n = 20, each block will have 3 residual blocks.
		self.X = x
		self.NUM_CLASSES = num_classes
		# Store the last layer of the network graph.
		self.out = None

		self.create()

	def create(self):

		conv1 = conv_layer(self.X, 3, 3, 16, name = 'conv1')
		self.out = conv1

		""" All residual blocks use zer-padding
		for shortcut connections """

		for i in range(self.NUM_CONV):
			resBlock2 = residual_block(self.out, 16, name = 'resBlock2_{}'.format(i + 1))
			self.out = resBlock2

		pool2 = max_pool(self.out, name = 'pool2')
		self.out = pool2

		for i in range(self.NUM_CONV):
			resBlock3 = residual_block(self.out, 32, name = 'resBlock3_{}'.format(i + 1))
			self.out = resBlock3

		pool3 = max_pool(self.out, name = 'pool3')
		self.out = pool3

		for i in range(self.NUM_CONV):
			resBlock4 = residual_block(self.out, 64, name = 'resBlock4_{}'.format(i + 1))
			self.out = resBlock4

		# Perform global average pooling to make spatial dimensions as 1x1
		global_pool = global_average(self.out, name = 'gap')
		self.out = global_pool

		flatten = tf.contrib.layers.flatten(self.out)
		fc5 = fc_layer(flatten, input_size = 64, output_size = self.NUM_CLASSES,
			relu = False, name = 'fc5')

		self.out = fc5
