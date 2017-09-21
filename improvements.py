# Good work. If in the future you want to adjust the parameters of the normalization you can use this general code:

def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [a, b]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    min_val = 0
    max_val = 255
    return a + ( ( (image_data - min_val)*(b - a) )/( max_val - min_val ) )
# The a and b are the lower and upper bounds respectively. The min_val and max_val are the current ranges of the data.
# For more information about why normalization is necessary in many types of machine learning problems (not just CNNs) check out this link:
# http://scikit-learn.org/dev/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py

# ** The fully_conn function creates a fully connected layer with a nonlinear activation.
# Nice work. Just for your information, a fully connected layer introduces a large number of new parameters to the model. I don't know if you've noticed, but this model takes a fairly long time to train and does not produce results even close to what a human can do. Recently, there has been a push to only have convolution layers in neural networks, and indeed the best results for this dataset have no fully connected layers. Check it out!

# I'm not saying that they shouldn't be included in all cases, but it's good to be conscious of the effect that fully connected layers have on training and running times.


# The conv_net function creates a convolutional model and returns the logits. Dropout should be applied to alt least one layer.
# Good work! Try this architecture if you have time. It's produced the best results i've seen so far:

def conv_net(x, keep_prob):
    conv_num_outputs = [32, 128, 256]
    conv_ksize = [(4, 4),(6,6), (8,8)]
    conv_strides = (1, 1)
    pool_ksize = (4, 4)
    pool_strides = (2, 2)
    num_outputs = 10


    conv1 = conv2d_maxpool(x, conv_num_outputs[0], conv_ksize[0], conv_strides, pool_ksize, pool_strides)
    drop1 = tf.nn.dropout(conv1, keep_prob=keep_prob)
    conv2 = conv2d_maxpool(drop1, conv_num_outputs[1], conv_ksize[1], conv_strides, pool_ksize, pool_strides)
    drop2 = tf.nn.dropout(conv2, keep_prob=keep_prob)
    conv3 = conv2d_maxpool(drop2, conv_num_outputs[2], conv_ksize[2], conv_strides, pool_ksize, pool_strides)
    drop3 = tf.nn.dropout(conv3, keep_prob=keep_prob)

    flatten1 = flatten(drop3)
    fc1 = fully_conn(flatten1, num_outputs)
    fc2 = fully_conn(fc1, num_outputs)


    out1 = output(fc2, num_outputs)

    return out1