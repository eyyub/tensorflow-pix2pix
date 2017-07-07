import numpy as np
import tensorflow as tf
from model.discriminator import Discriminator
from model.generator import Generator

class Pix2pix(object):
    def __init__(self, width, height, ichan, ochan, l1_weight=100., lr=0.0002, beta1=0.5):
        """
            width: image width in pixel.
            height: image height in pixel.
            ichan: number of channels used by input images.
            ochan: number of channels used by output images.
            l1_weight: L1 loss weight.
            lr: learning rate for ADAM optimizer.
            beta1: beta1 parameter for ADAM optimizer.
        """
        self._is_training = tf.placeholder(tf.bool)

        self._g_inputs = tf.placeholder(tf.float32, [None, width, height, ichan])
        self._d_inputs_a = tf.placeholder(tf.float32, [None, width, height, ichan])
        self._d_inputs_b = tf.placeholder(tf.float32, [None, width, height, ochan])
        self._g = Generator(self._g_inputs, self._is_training, ochan)
        self._real_d = Discriminator(tf.concat([self._d_inputs_a, self._d_inputs_b], axis=3), self._is_training)
        self._fake_d = Discriminator(tf.concat([self._d_inputs_a, self._g._decoder['cl9']['fmap']], axis=3), self._is_training, reuse=True)

        self._g_loss = -tf.reduce_mean(tf.log(self._fake_d._discriminator['l5']['fmap'])) + l1_weight * tf.reduce_mean(tf.abs(self._d_inputs_b - self._g._decoder['cl9']['fmap']))
        self._d_loss = -tf.reduce_mean(tf.log(self._real_d._discriminator['l5']['fmap']) + tf.log(1.0 - self._fake_d._discriminator['l5']['fmap']))

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self._g_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._g_loss,
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self._d_train_step = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self._d_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))

    def train_step(self, sess, g_inputs, d_inputs_a, d_inputs_b, is_training=True):
        _, dloss_curr = sess.run([self._d_train_step, self._d_loss],
            feed_dict={self._d_inputs_a : d_inputs_a, self._d_inputs_b : d_inputs_b, self._g_inputs : g_inputs, self._is_training : is_training})
        _, gloss_curr = sess.run([self._g_train_step, self._g_loss],
                feed_dict={self._g_inputs : g_inputs, self._d_inputs_a : d_inputs_a,   self._d_inputs_b : d_inputs_b,self._is_training : is_training})
        return (gloss_curr, dloss_curr)

    def sample_generator(self, sess, g_inputs, is_training=False):
        return sess.run(self._g._decoder['cl9']['fmap'], feed_dict={self._g_inputs : g_inputs, self._is_training : is_training})
