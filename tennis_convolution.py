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
num_input = 16 # MNIST data input (img shape: 28*28)
dropout = 0.75 # Dropout, probability to keep units
num_outputs = 2 # digits composing the score we want to predict

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
    # Data input is a 1-D vector of 16 features (4x4 matrix)
    # Reshape to match [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 4, 4, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, score prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    # 2x2 conv, 1 input, 4 outputs
    'wc1': tf.Variable(tf.random_normal([2, 2, 1, 4])),
    # 2x2 conv, 4 inputs, 8 outputs
    'wc2': tf.Variable(tf.random_normal([2, 2, 4, 8])),
    # 4x4 conv, 8 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([4, 4, 8, 64])),
    # fully connected, 8*8*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*16, 1024])),
    # 1024 inputs, 2 outputs (score prediction)
    'out': tf.Variable(tf.random_normal([1024, num_outputs]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([4])),
    'bc2': tf.Variable(tf.random_normal([8])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_outputs]))
}

logits = conv_net(X, weights, biases, keep_prob)

# Mean squared error
cost_op = tf.reduce_sum(tf.pow(logits-Y, 2))/(2*num_input)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost_op)

# Evaluate model
correct_pred = tf.equal(logits, Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        for batch in split_batch(data, batch_size):
            batch_x = [elem[:-2] for elem in batch]
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
