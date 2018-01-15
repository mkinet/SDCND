### Define your architecture here.
### Feel free to use as many code cells as needed.

graph = tf.Graph()

with graph.as_default():
    def LeNet(x, keep_prob, l2, is_training):
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x16. Activation = relu
        patch_size1 = 5
        num_channels1 = 1
        depth1 = 16
        stride1 = 1
        weights1 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, num_channels1, depth1],
                                                   stddev=np.sqrt(2 / (5 * 5 * 1))))
        biases1 = tf.Variable(tf.zeros([depth1]))
        conv1 = tf.nn.conv2d(x, weights1, [1, stride1, stride1, 1], padding='VALID')
        relu1 = tf.nn.dropout(tf.nn.relu(tf.add(conv1, biases1)), keep_prob)
        # Layer 2: Convolutional. Output = 24x24x32. Activation = relu
        patch_size2 = 5
        num_channels2 = depth1
        depth2 = 32
        stride2 = 1
        weights2 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, num_channels2, depth2],
                                                   stddev=np.sqrt(2 / (5 * 5 * 16))))
        biases2 = tf.Variable(tf.zeros([depth2]))
        conv2 = tf.nn.conv2d(relu1, weights2, [1, stride2, stride2, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.add(conv2, biases2))
        # Pooling. Input = 24x24x32. Output = 12x12x32.
        pool2 = tf.nn.dropout(tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID'), keep_prob)

        # Layer 3: Convolutional. Output = 8x8x64. Activation = relu
        patch_size3 = 5
        num_channels3 = 32
        depth3 = 64
        stride3 = 1
        weights3 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, num_channels3, depth3],
                                                   stddev=np.sqrt(2 / (5 * 5 * 32))))
        biases3 = tf.Variable(tf.zeros([depth3]))
        conv3 = tf.nn.conv2d(pool2, weights3, [1, stride3, stride3, 1], padding='VALID')
        relu3 = tf.nn.relu(tf.add(conv3, biases3))
        # Pooling. Input = 8x8x64. Output = 4x4*64
        pool3 = tf.nn.dropout(tf.nn.max_pool(relu3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID'), keep_prob)

        # TODO: Flatten. Input = 4x4x64. Output = 1024.
        flat = flatten(pool3)
        # TODO: Layer 4: Fully Connected. Input = 1024. Output = 256.
        weights4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=np.sqrt(2 / (1024))))
        biases4 = tf.Variable(tf.zeros([512]))
        fully4 = tf.add(tf.matmul(flat, weights4), biases4)
        # Activation.
        relu4 = tf.nn.relu(fully4)

        # Layer 5: Fully Connected. Input = 256. Output = 128.
        weights5 = tf.Variable(tf.truncated_normal([512, 128], stddev=np.sqrt(2 / (512))))
        biases5 = tf.Variable(tf.zeros([128]))
        fully5 = tf.add(tf.matmul(relu4, weights5), biases5)

        # Activation.
        relu5 = tf.nn.relu(fully5)

        # Layer 6: Fully Connected. Input = 128. Output = 43.
        weights6 = tf.Variable(tf.truncated_normal([128, 43], stddev=np.sqrt(2 / (128))))
        biases6 = tf.Variable(tf.zeros([43]))
        logits = tf.add(tf.matmul(relu5, weights6), biases6)
        l2_loss = l2 * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)
                        + tf.nn.l2_loss(weights4) + tf.nn.l2_loss(weights5) + tf.nn.l2_loss(weights6))
        return logits, l2_loss


    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    p = tf.placeholder(tf.float32, ())
    is_training = tf.placeholder(tf.bool, ())
    one_hot_y = tf.one_hot(y, 43)

    rate = 0.0005
    l2 = 5e-4

    logits, l2_loss = LeNet(x, p, l2, is_training)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

    loss_operation = tf.reduce_mean(cross_entropy) + l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # class probabilities
    # top5_classes = tf.nn.top_k(tf.nn.softmax(logits), k=5)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()