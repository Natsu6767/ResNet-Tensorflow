import tensorflow as tf

def conv_layer(x, filter_height, filter_width,
	num_filters, name, stride = 1, padding = 'SAME'):

	"""Create a convolution layer."""
	
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])

	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases of the conv layer
		W = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [num_filters], initializer = tf.constant_initializer(0.0))

		# Perform convolution.
		conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
		# Add the biases.
		z = tf.nn.bias_add(conv, b)

		# Permorm batch normalization
		batch_norm = tf.layers.batch_normalization(z, axis = 1, beta_initializer = tf.constant_initializer(0.0),
			gamma_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
		# Apply ReLu non linearity.
		out = tf.nn.relu(batch_norm)

		return out

def max_pool(x, name, filter_height = 2, filter_width = 2,
	stride = 2, padding = 'VALID'):

	"""Create a max pooling layer."""

	return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
		strides = [1, stride, stride, 1], padding = padding, name = name)

def global_average(x, name):

	"""Create a global average pooling layer"""

	filter_hw = int(x.get_shape()[1])
	gap = tf.nn.avg_pool(x, ksize = [1, filter_hw, filter_hw, 1],
		strides = [1, 1, 1, 1], padding = 'VALID', name = name)

	return gap

def fc_layer(x, input_size, output_size, name, relu = True):

	"""Create a fully connected layer."""
	
	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases.
		W = tf.get_variable('weights', shape = [input_size, output_size],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))

		# Matrix multiply weights and inputs and add biases.
		z = tf.nn.bias_add(tf.matmul(x, W), b)

		if relu:
			# Apply ReLu non linearity.
			a = tf.nn.relu(z)
			return a

		else:
			return z