import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 2x upsample via transposed convolution ('deconvolution')
def deconv2d2x(x, W, output_shape):
  return tf.nn.conv2d_transpose(
          x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')

batch_size = tf.placeholder(tf.int32, [], 'batch_size')

x = tf.placeholder(tf.float32, [None, 10], 'x')

with tf.variable_scope('fc1'):
    W_fc1 = weight_variable([10, 1024])
    b_fc1 = bias_variable([1024])
    x_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

with tf.variable_scope('fc2'):
    W_fc2 = weight_variable([1024, 7 * 7 * 64])
    b_fc2 = bias_variable([7*7*64])
    x_fc2_flat = tf.matmul(x_fc1, W_fc2) + b_fc2
    x_fc2 = tf.nn.relu(tf.reshape(x_fc2_flat, [-1, 7, 7, 64]))

# Deconv 1
with tf.variable_scope('deconv1'):
    W_conv1 = weight_variable([5, 5, 32, 64])
    b_conv1 = bias_variable([32])
    oshape1 = [batch_size, 14, 14, 32]
    h_conv1 = deconv2d2x(x_fc2, W_conv1, oshape1) + b_conv1

# Deconv 2
with tf.variable_scope('deconv2'):
    W_conv2 = weight_variable([5, 5, 1, 32])
    b_conv2 = bias_variable([1])
    oshape2 = [batch_size, 28, 28, 1]
    h_conv2 = deconv2d2x(h_conv1, W_conv2, oshape2) + b_conv2

# Final image!
y = tf.reshape(h_conv2, [-1, 784])

# Cost function
y_ = tf.placeholder(tf.float32, [None, 784], name='y_')
cost = tf.losses.mean_squared_error(y_, y)

# Optional: Enforce sparse activations (note: not normalized
#           for batch size)
# Comment out:
#cost2 = tf.reduce_mean(x_fc2)
#cost = cost + cost2

learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    N = 50
    batch = mnist.train.next_batch(N)
    batch_x = batch[1]
    batch_y = batch[0]
    batch_cost, _ = sess.run([cost, train_step], feed_dict={
            x: batch_x,
            y_: batch_y,
            learning_rate: 1e-4,
            batch_size: N})
    print ('batch %s cost: %s' % (i, batch_cost))

def gen_image(one_hot):
#    '''Inference: create single image from one-hot vector'''
    y_img = sess.run([y], feed_dict={x: [one_hot], batch_size:1})
    return y_img[0].reshape((28, 28))

def show(img):
    plt.figure()
    plt.imshow(img)
    
def single_digit(digit):
    '''Make a one-hot vector to represent a single digit'''
    one_hot = np.zeros([10])
    one_hot[digit] = 1.0
    return one_hot

def blend_digits(digit1, digit2, amount):
    '''Return a one-hot vector average of digit1 and digit2'''
    one_hot = np.zeros([10])
    one_hot[digit1] = 1.0 - amount
    one_hot[digit2] = amount
    return one_hot

# Demo
show(gen_image(blend_digits(1,2,0.5)))
for d in range(10):
    show(gen_image(single_digit(d)))

# Generate an image sequence by blending digits from 0 to 10
frame = 0
N = 24
for d in range(10):
    d_next = (d + 1) % 10
    for i in range(N):
        one_hot = blend_digits(d,d_next,i/(float(N) - 1))
        img = gen_image(one_hot)
        img = misc.imresize(img, (100, 100), interp='nearest')
        misc.imsave('/tmp/mnist-out/img.%03d.png' % frame, img)
        frame += 1