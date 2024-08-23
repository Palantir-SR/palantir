import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class PALANTIR_S():
    def __init__(self, num_blocks, num_filters, scale, upsample_type='deconv'):
        assert(upsample_type == 'deconv' or upsample_type == 'subpixel')

        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.scale = scale
        self.upsample_type = upsample_type
        self.conv_idx = 0

        #name
        self.name = self.__class__.__name__
        self.name += '_B{}'.format(self.num_blocks)
        self.name += '_F{}'.format(self.num_filters)
        self.name += '_S{}'.format(self.scale)
        self.name += '_{}'.format(upsample_type)

    def _conv_name(self):
        if self.conv_idx== 0:
            name = 'conv2d'
        else:
            name = 'conv2d_{}'.format(self.conv_idx)
        self.conv_idx += 1
        return name

    def _residual_block(self, x_in, num_filters):
        x = layers.Conv2D(num_filters, 3, padding='same', activation='relu', name=self._conv_name())(x_in)
        x = layers.Conv2D(num_filters, 3, padding='same', name=self._conv_name())(x)
        x = layers.Add()([x_in, x])
        return x

    def build(self, output_shape=None, apply_clip=False, shape=(None, None, 3)):
        x_in = layers.Input(shape=shape)

        x = b = layers.Conv2D(self.num_filters, 3, padding='same', name=self._conv_name())(x_in)

        for i in range(self.num_blocks):
            b = self._residual_block(b, self.num_filters)
        b = layers.Conv2D(self.num_filters, 3, padding='same', name=self._conv_name())(b)
        x = layers.Add()([x, b])

        if self.upsample_type == 'deconv':
            if self.scale in [2, 3, 4]:
                x = layers.Conv2DTranspose(3, 5, self.scale, padding='same')(x)
            else:
                raise NotImplementedError
        elif self.upsample_type == 'subpixel':
            if self.scale == 2 or self.scale ==  3:
                x = layers.Conv2D(3 * (self.scale ** 2), 3, padding='same', name=self._conv_name())(x)
                x = layers.Lambda(lambda x:tf.nn.depth_to_space(x, self.scale))(inputs=x)
            elif self.scale == 4:
                x = layers.Conv2D(3 * 4, 3, padding='same', name=self._conv_name())(x)
                x = layers.Lambda(lambda x:tf.nn.depth_to_space(x, 2))(inputs=x)
                x = layers.Conv2D(3 * 4, 3, padding='same', name=self._conv_name())(x)
                x = layers.Lambda(lambda x:tf.nn.depth_to_space(x, 2))(inputs=x)
            else:
                raise NotImplementedError

        if output_shape is not None:
            x = tf.image.resize_bilinear(x, (output_shape[1], output_shape[2]), half_pixel_centers=True) # shape: [batch, height, width, channel]

        if apply_clip is True:
            x = tf.minimum(x, 255)
            x = tf.maximum(x, 0)

        model = Model(inputs=x_in, outputs=x, name=self.name)

        return model

    def load(self, checkpoint_dir):
        model = self.build()
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=1)
        checkpoint_path = checkpoint_manager.latest_checkpoint
        assert(checkpoint_path is not None)
        checkpoint.restore(checkpoint_path).expect_partial()

        return checkpoint