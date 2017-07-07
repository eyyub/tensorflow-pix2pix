import numpy as np
import tensorflow as tf
from .utils import get_shape, batch_norm, lkrelu

# U-Net Generator
class Generator(object):
    def __init__(self, inputs, is_training, ochan, stddev=0.02, center=True, scale=True, reuse=None):
        self._is_training = is_training
        self._stddev = stddev
        self._ochan = ochan
        with tf.variable_scope('G', initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._center = center
            self._scale = scale
            self._prob = 0.5 # constant from pix2pix paper
            self._inputs = inputs
            self._encoder = self._build_encoder(inputs)
            self._decoder = self._build_decoder(self._encoder)

    def _build_encoder_layer(self, name, inputs, k, bn=True, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name):
            layer['filters'] = tf.get_variable('filters', [4, 4, get_shape(inputs)[-1], k])
            layer['conv'] = tf.nn.conv2d(inputs, layer['filters'], strides=[1, 2, 2, 1], padding='SAME')
            layer['bn'] = batch_norm(layer['conv'], center=self._center, scale=self._scale, training=self._is_training) if bn else layer['conv']
            layer['dropout'] = tf.nn.dropout(layer['bn'], self._prob) if use_dropout else layer['bn']
            layer['fmap'] = lkrelu(layer['dropout'], slope=0.2)
        return layer

    def _build_encoder(self, inputs):
        encoder = dict()

        # C64-C128-C256-C512-C512-C512-C512-C512
        with tf.variable_scope('encoder'):
            encoder['l1'] = self._build_encoder_layer('l1', inputs, 64, bn=False)
            encoder['l2'] = self._build_encoder_layer('l2', encoder['l1']['fmap'], 128)
            encoder['l3'] = self._build_encoder_layer('l3', encoder['l2']['fmap'], 256)
            encoder['l4'] = self._build_encoder_layer('l4', encoder['l3']['fmap'], 512)
            encoder['l5'] = self._build_encoder_layer('l5', encoder['l4']['fmap'], 512)
            encoder['l6'] = self._build_encoder_layer('l6', encoder['l5']['fmap'], 512)
            encoder['l7'] = self._build_encoder_layer('l7', encoder['l6']['fmap'], 512)
            encoder['l8'] = self._build_encoder_layer('l8', encoder['l7']['fmap'], 512)
        return encoder

    def _build_decoder_layer(self, name, inputs, output_shape_from,use_dropout=False):
        layer = dict()

        with tf.variable_scope(name):
            output_shape = tf.shape(output_shape_from)
            layer['filters'] = tf.get_variable('filters', [4, 4, get_shape(output_shape_from)[-1], get_shape(inputs)[-1]])
            layer['conv'] = tf.nn.conv2d_transpose(inputs, layer['filters'], output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')
            layer['bn'] = batch_norm(tf.reshape(layer['conv'], output_shape), center=self._center, scale=self._scale, training=self._is_training)
            layer['dropout'] = tf.nn.dropout(layer['bn'], self._prob) if use_dropout else layer['bn']
            layer['fmap'] = tf.nn.relu(layer['dropout'])
        return layer

    def _build_decoder(self, encoder):
        decoder = dict()

        # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        with tf.variable_scope('decoder'): # U-Net
            decoder['dl1'] = self._build_decoder_layer('dl1', encoder['l8']['fmap'], output_shape_from=encoder['l7']['fmap'], use_dropout=True)

            # fmap_concat represent skip connections
            fmap_concat = tf.concat([decoder['dl1']['fmap'], encoder['l7']['fmap']], axis=3)
            decoder['dl2'] = self._build_decoder_layer('dl2', fmap_concat, output_shape_from=encoder['l6']['fmap'], use_dropout=True)

            fmap_concat = tf.concat([decoder['dl2']['fmap'], encoder['l6']['fmap']], axis=3)
            decoder['dl3'] = self._build_decoder_layer('dl3', fmap_concat, output_shape_from=encoder['l5']['fmap'], use_dropout=True)

            fmap_concat = tf.concat([decoder['dl3']['fmap'], encoder['l5']['fmap']], axis=3)
            decoder['dl4'] = self._build_decoder_layer('dl4', fmap_concat, output_shape_from=encoder['l4']['fmap'])

            fmap_concat = tf.concat([decoder['dl4']['fmap'], encoder['l4']['fmap']], axis=3)
            decoder['dl5'] = self._build_decoder_layer('dl5', fmap_concat, output_shape_from=encoder['l3']['fmap'])

            fmap_concat = tf.concat([decoder['dl5']['fmap'], encoder['l3']['fmap']], axis=3)
            decoder['dl6'] = self._build_decoder_layer('dl6', fmap_concat, output_shape_from=encoder['l2']['fmap'])

            fmap_concat = tf.concat([decoder['dl6']['fmap'], encoder['l2']['fmap']], axis=3)
            decoder['dl7'] = self._build_decoder_layer('dl7', fmap_concat, output_shape_from=encoder['l1']['fmap'])

            fmap_concat = tf.concat([decoder['dl7']['fmap'], encoder['l1']['fmap']], axis=3)
            decoder['dl8'] = self._build_decoder_layer('dl8', fmap_concat, output_shape_from=self._inputs)

            with tf.variable_scope('cl9'):
                cl9 = dict()
                cl9['filters'] = tf.get_variable('filters', [4, 4, get_shape(decoder['dl8']['fmap'])[-1], self._ochan])
                cl9['conv'] =  tf.nn.conv2d(decoder['dl8']['fmap'], cl9['filters'], strides=[1, 1, 1, 1], padding='SAME')
                cl9['fmap'] = tf.nn.tanh(cl9['conv'])
                decoder['cl9'] = cl9
        return decoder
