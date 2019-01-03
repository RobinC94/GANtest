#!/usr/bin/python
# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from termcolor import cprint

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("z_dim", 100, "Dimension of z")
flags.DEFINE_integer("batch_size", 50, "Batch size")
flags.DEFINE_integer("pre_epoches", 300, "Pre-train epoches")
flags.DEFINE_integer("epoches", 100000, "Epoches")
flags.DEFINE_float("lr", 1e-4, "Learning rate")


def discriminator(images):
    ## Small CNN with 2 conv and 2 fc
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):

        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.zeros_initializer())
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.zeros_initializer())
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w3 = tf.get_variable('d_w3', [7*7*64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.zeros_initializer())
        d3 = tf.reshape(d2, [-1, 7*7*64])
        d3 = tf.matmul(d3, d_w3) + d_b3
        d3 = tf.nn.relu(d3)

        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.zeros_initializer())
        d4 = tf.matmul(d3, d_w4) + d_b4

    return d4


def generator(z, batch_size, z_dim):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')

        # Generate 50 features
        g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b2 = tf.get_variable('g_b2', [z_dim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g1, filter=g_w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + g_b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])

        # Generate 25 features
        g_w3 = tf.get_variable('g_w3', [3, 3, z_dim / 2, z_dim/4], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b3 = tf.get_variable('g_b3', [z_dim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, filter=g_w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + g_b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])

        # Final convolution with one output channel
        g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g3, filter=g_w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + g_b4
        g4 = tf.sigmoid(g4)

    # batch x 28 x 28 x 1
    return g4


def get_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/")
    #display_img(mnist.train.next_batch(1)[0])
    return mnist


def display_img(sample_image):
    print(sample_image.shape)

    sample_image = sample_image.reshape([28, 28])
    plt.imshow(sample_image, cmap='Greys')
    plt.show()


def generate_image(sess, z_placeholder, z_dim):
    z_batch = np.random.normal(0, 1, [1, z_dim])
    generated_image = generator(z_placeholder, 1, z_dim)

    img = sess.run(generated_image,
                      feed_dict={z_placeholder: z_batch})
    plt.imshow(img[0].reshape([28, 28]), cmap='Greys')
    plt.show()

    result = discriminator(x_placeholder)
    estimate = sess.run(result, feed_dict={x_placeholder: img.reshape([1, 28, 28, 1])})[0]
    return estimate


def get_vars():
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    print([v.name for v in d_vars])
    print([v.name for v in g_vars])
    return d_vars, g_vars


if __name__ == "__main__":
    mnist = get_data()

    batch_size = FLAGS.batch_size
    z_dimensions = FLAGS.z_dim

    tf.reset_default_graph()
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
    x_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_placeholder')

    Gz = generator(z_placeholder, batch_size, z_dimensions)
    Dx = discriminator(x_placeholder)
    Dg = discriminator(Gz)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

    d_vars, g_vars = get_vars()
    learning_rate = FLAGS.lr

    # Train the discriminator
    d_trainer_real = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_real, var_list=d_vars)
    d_trainer_fake = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_fake, var_list=d_vars)

    # Train the generator
    g_trainer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    tf.summary.scalar('Generator loss', g_loss)
    tf.summary.scalar('Discriminator loss real', d_loss_real)
    tf.summary.scalar('Discriminator loss fake', d_loss_fake)

    with tf.Session() as sess:

        images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
        tf.summary.image('Generator images', images_for_tensorboard, 5)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
        writer = tf.summary.FileWriter(logdir, sess.graph)

        sess.run(tf.global_variables_initializer())

        # Pre-train discriminator
        cprint("Start Warming-Up...", 'green')
        for i in range(FLAGS.pre_epoches):

            z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
            real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   feed_dict={x_placeholder: real_image_batch,
                                                              z_placeholder: z_batch})

            if i % 100 == 0:
                print("Iteration: ", i, "  D loss real: ", dLossReal, "  fake: ", dLossFake)

        # Train generator and discriminator together
        cprint("Now Start Training...", 'green')
        for i in range(FLAGS.epoches):
            real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])

            # Train discriminator on both real and fake images
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   feed_dict={x_placeholder: real_image_batch,
                                                              z_placeholder: z_batch})

            # Train generator
            z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
            _, gLoss = sess.run([g_trainer, g_loss], feed_dict={z_placeholder: z_batch})

            if i % 10 == 0:
                # Update TensorBoard with summary statistics
                z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
                summary = sess.run(merged, feed_dict={z_placeholder: z_batch, x_placeholder: real_image_batch})
                writer.add_summary(summary, i)
                print("Iteration: ", i, "  D loss real: ", dLossReal, "  fake: ", dLossFake, "  G loss: ", gLoss)

            if i % 100 == 0:
                # Every 100 iterations, show a generated image
                estimate = generate_image(sess, z_placeholder, z_dimensions)
                print("Estimate:", estimate)




