import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
import numpy as np

import config

class DeepLab():
    '''
    网络的输入 [batch_size, 256, 256, 4]
    网络的输出 [batch_size, 256, 256, 1]
    预测[164, 164]大小的图片中每一个像素点的分类
    '''
    def __init__(self):
        self.batch_size = config.batch_size
        self.images = tf.placeholder(tf.float32, [None, 256, 256, 4])
        self.labels = tf.placeholder(tf.float32, [None, 256, 256, 1])
        self.batch_size = tf.shape(self.images)[0]
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits = self.build_model(self.images)
        self.loss = self.loss_layer(self.logits, self.labels)
        tf.summary.scalar('loss', self.loss)

        self.predits = tf.cast(tf.greater(self.logits, 0.5), dtype=tf.float32)
        self.dice_coef = self.dice_coefficient(self.predits, self.labels)
        tf.summary.scalar('dice coefficient', self.dice_coef)

    def build_model(self, input):
        print(input)
        input_size = tf.shape(input)[1:3]

        '''
        Encoder
        '''
        _, endpoints = resnet_v2.resnet_v2_101(inputs=input, num_classes=None, is_training=True, global_pool=False, output_stride=16)
        net = endpoints['resnet_v2_101' + '/block4']
        print(net)
        feature_shape = tf.shape(net)[1:3]

        # Atrous spatial pyramid pooling

        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18)
        conv_1x1 = layers_lib.conv2d(net, 256, [1, 1], stride=1, scope="conv_1x1")
        print(conv_1x1)
        conv_3x3_1 = layers_lib.conv2d(net, 256, [3, 3], stride=1, rate=6, scope='conv_3x3_6')
        print(conv_3x3_1)
        conv_3x3_2 = layers_lib.conv2d(net, 256, [3, 3], stride=1, rate=12, scope='conv_3x3_12')
        print(conv_3x3_2)

        # conv_1x1 = self.conv_2d(net, [1, 1, 2048, 256], name='conv_1x1', padding='SAME')
        # conv_3x3_1 = self.conv_2d(net, [2, 2, 2048, 256], dilations=[1, 6, 6, 1], padding='SAME', name='conv_3x3_6')
        # conv_3x3_2 = self.conv_2d(net, [2, 2, 2048, 256], dilations=[1, 12, 12, 1], padding='SAME', name='conv_3x3_12')
        # conv_3x3_3 = self.conv_2d(net, [2, 2, 2048, 256], dilations=[1, 18, 18, 1], name='conv_3x3_18')
        # (b) the inage-level features
        # (b.1) global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='global_average_pooling', keepdims=True)
        # (b.2) 1x1 convolution with 256 filters( and batch normalization)
        image_level_features = layers_lib.conv2d(image_level_features, 256, [1, 1], stride=1, scope='image-pooling')
        # image_level_features = self.conv_2d(image_level_features, [1, 1, 2048, 256], name='image-pooling')
        # (b.3) bilinearly upsample features
        image_level_features = tf.image.resize_bilinear(image_level_features, feature_shape, name='upsample')
        print(image_level_features)
        # (c) concat 4 kinds of features
        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, image_level_features], axis=3, name='encoder_concat')
        encoder_output = layers_lib.conv2d(net, 256, [1, 1], stride=1, scope='encoder-output')
        # encoder_output = self.conv_2d(net, [1, 1, 1024, 256], name='encoder-output')
        print(encoder_output)

        '''
        Decoder
        '''

        # low-level features
        low_level_features = endpoints['resnet_v2_101' + '/block1/unit_3/bottleneck_v2/conv1']
        print(low_level_features)
        low_level_features = self.conv_2d(low_level_features, [1, 1, 64, 48], padding='SAME',name='low-level-features-conv')
        low_level_features_size = tf.shape(low_level_features)[1:3]

        # up-sampling logits
        net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
        net = tf.concat([net, low_level_features], axis=3, name='decoder_concat')
        print(net)
        net = self.conv_2d(net, [3, 3, 304, 256], name='decoder-conv-1')
        print(net)
        net = self.conv_2d(net, [3, 3, 256, 256], name='decoder-conv-2')
        print(net)
        net = self.conv_2d(net, [1, 1, 256, 1], ac=tf.nn.sigmoid, name='decoder-conv-3')
        print(net)
        # net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
        # net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
        # net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
        logits = tf.image.resize_bilinear(net, input_size, name='upsample_2')
        print(logits)

        return logits

    def loss_layer(self, logits, labels):
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(tf.square(logits - labels))
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits + 1e-6)
            # loss = - tf.reduce_mean((1 - labels) * tf.log(1 - logits + 1e-6) + labels * tf.log(logits + 1e-6))
            # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits + 1e-6))
            tf.losses.add_loss(loss)
            print('loss = ', loss)
        return loss

    def dice_coefficient(self, predicts, labels):
        intersection = tf.reduce_sum(predicts * labels)
        union = tf.reduce_sum(predicts) + tf.reduce_sum(labels)
        dice_coef = (2.0 * intersection) / (union + 1.0)
        return dice_coef

    def conv_2d(self, input, kernel_shape, strides=[1, 1, 1, 1], dilations = [1, 1, 1, 1], ac=tf.nn.relu, padding="VALID", name=None):
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", shape=kernel_shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))
            kernel = tf.nn.dropout(kernel, keep_prob=self.keep_prob)
            bias = tf.get_variable("bias", shape=[kernel_shape[-1]], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input, kernel, strides, padding=padding, dilations=dilations)
            conv = tf.nn.bias_add(conv, bias)
            conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.8, is_training=True)
            # conv = tf.layers.batch_normalization(conv)
            conv = ac(conv)
        return conv

    def conv_2d_transpose(self, input, kernel_shape, output_shape, strides=[1, 2, 2, 1], ac=tf.nn.relu, padding="VALID",
                          name=None):

        with tf.variable_scope(name):
            kernel = tf.random_normal(shape=kernel_shape)
            output = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=strides, padding=padding)            # kernel = tf.get_variable("weights", shape=kernel_shape, dtype=tf.float32,
            # output = tf.layers.batch_normalization(output)
            output = tf.contrib.layers.batch_norm(output, center=True, scale=True, decay=0.8, is_training=True)
            output = ac(output)
            #                          initializer=tf.constant_initializer(0.01))
            # kernel = tf.nn.dropout(kernel, keep_prob=self.keep_prob)
            # bias = tf.get_variable("bias", shape=[kernel_shape[-2]], dtype=tf.float32,
            #                        initializer=tf.constant_initializer(0.0))
            # conv = tf.nn.conv3d_transpose(input, kernel, output_shape, strides, padding=padding)
            # conv = tf.nn.bias_add(conv, bias)
            # conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.8, is_training=True)
            # conv = ac(conv)
        return output


# with tf.variable_scope('test'):
#     x1 = tf.placeholder(tf.float32, [None, 28, 28, 28, 1])
#     x2 = tf.placeholder(tf.float32, [None, 28, 28, 28, 2])
#     y = tf.concat([x1, x2], 4)
#     print(y)
    # print(imgs)
    # net = conv_2d(imgs, [3, 3, 3, 1, 10], padding='VALID', name='conv_1')
    # print(net)
    # net = conv_2d_transpose(net, [2, 2, 2, 1, 10], [-1, 42, 42, 42, 1], padding='VALID', name='deconv_1')
    # net = tf.nn.conv2d_transpose(net, [2, 2, 10, 1], [1, 1, 1, 1], padding='SAME')
    # print(net)

# net = DeepLab()
# print(net.loss)
