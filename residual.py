import tensorflow as tf

from layers import conv_layer

def residual_block(x, out_channels, projection = False, name = 'residual'):

	"""Create a Residual Block with two conv layers"""

	# Get the input channels
	input_channels = int(x.get_shape()[-1])

	conv1 = conv_layer(x, 3, 3, out_channels, stride = 1, name = '{}_conv1'.format(name))
	conv2 = conv_layer(conv1, 3, 3 out_channels, stride = 1, name = '{}_conv2'.format(name))

	if input_channels != out_channels:
		if projection:
			# Option B: Projection Shortcut
			# This introduces extra parameters.
			# Stride has been set as 2 as zero-padding has been used during implementing
			# conv_layer in layers.py. If the stride is different from 2, the spatial dimensions
			# will change (reason 1x1 filters are used).
			shortcut = conv_layer(x, 1, 1, out_channels, stride = 2, name = '{}_shortcut'.format(name))
		else:
			# Option A: Zero-Padding
			# This method doesn't introduce any extra parameters.
			shortcut = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, out_channels - input_channels]])
	else:
		shortcut = x


	out = conv2 + shortcut

	return out
