import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import matplotlib.pyplot as plt

# TODO: Load traffic signs data.
# Load pickled data
training_file = 'train.p'
nb_classes = 43
with open(training_file, mode='rb') as f:
    data = pickle.load(f)

X_0, y_0 = data['features'], data['labels']
X_0 = X_0[:int(4096//0.8)]
y_0 = y_0[:int(4096//0.8)]
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_0, y_0, test_size=0.2)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
#
# # TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape,stddev=0.01))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.add(tf.matmul(fc7,fc8W),fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, 43)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

# Accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.

EPOCHS = 10
BATCH_SIZE = 64
train_accuracy = []
valid_accuracy = []

sess = tf.Session()
sess.run(init)
sess.run(tf.global_variables_initializer())
num_examples = len(X_train)
print('There are %d training examples' %num_examples)
print("Training...")
print()
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):

        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

    acc_t = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
    acc_v = sess.run(accuracy_operation, feed_dict={x: X_valid, y: y_valid})
    print("EPOCH {} ...".format(i + 1))
    print("MiniBatch Accuracy = {:.3f}".format(acc_t))
    print("Validation Accuracy = {:.3f}".format(acc_v))
    print()
    train_accuracy.append(acc_t)
    valid_accuracy.append(acc_v)


plt.figure()
plt.plot(np.arange(EPOCHS), train_accuracy, 'r', label='Training')
plt.plot(np.arange(EPOCHS), valid_accuracy, 'g', label='Validation')
plt.legend()
plt.show()