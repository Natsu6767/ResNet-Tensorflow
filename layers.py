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
		batch_norm = tf.layers.batch_normalization(z, beta_initializer = tf.constant_initializer(0.0),
			gamma_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
		# Apply ReLu non linearity.
		out = tf.nn.relu(batch_norm)

		return out