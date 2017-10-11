'''
Predicting the outcome of a tennis match using convolutional neural networks for regression

CNN trained to predict the final score (0-1, 2-1, etc.)

Input matrix is a 4x4:

| Player 1 Rank | Player 2 Rank | Avg. Service Speed P1 | Avg. Service Speed P2
| Surface wins ratio Player 1 (3 values)                | Time Elapsed since last match P1
| Surface wins ratio Player 2 (3 values)                | Time Elapsed since last match P2
| Court         | Surface       | Round                 | Series

Where Court, Surface, Round and Series are integers
'''
import os

import pandas as pd
import tensorflow as tf

from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('data/match_matrix.csv', delimiter=', ', engine='python').dropna(axis=0)

le = preprocessing.LabelEncoder()
to_encode = ['Court', 'Surface', 'Round', 'Series']
for col in to_encode:
    le.fit(data[col])
    data[col] = le.transform(data[col])

data.drop(['Winner', 'Loser'], inplace=True, axis=1)
data = data.values

# batch(data, batch_size)
# utility function to generate batches
def split_batch(iterable, n=1):
    l = len(iterable)
    batch_size = l//n
    for ndx in range(0, l, n):
        if ndx + n > l:
            return
        yield iterable[ndx:(ndx + n)]

# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 16 # input, 4x4 matrix
dropout = 0.75 # Dropout, probability to keep units
num_outputs = 2 

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_outputs])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME'
    )


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 4, 4, 1])

    # Convolution Layers and max-poolings
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = maxpool2d(conv5, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([2, 2, 1, 4])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([2, 2, 4, 8])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([2, 2, 8, 16])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([2, 2, 16, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc5': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*32, 1024])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([1024, 256])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, num_outputs]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([4])),
    'bc2': tf.Variable(tf.random_normal([8])),
    'bc3': tf.Variable(tf.random_normal([16])),
    'bc4': tf.Variable(tf.random_normal([32])),
    'bc5': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([num_outputs]))
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Mean squared error
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        for batch in split_batch(data, batch_size):
            batch_x = [elem[:-4] for elem in batch]
            batch_y = [elem[-2:] for elem in batch]
            # Run optimization op
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch cost and accuracy
            loss, acc = sess.run([cost_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Cost= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
