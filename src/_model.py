import tensorflow as tf
from six.moves import xrange
import string

from ._default import *
from ._utils import round_filter, round_repeats, _preprocess_padding, conv_output_length


class DepthWiseConv3D(tf.keras.layers.Conv3D):
    def __init__(self,
                 kernel_size, strides=(1, 1, 1), padding='valid', depth_multiplier=1, groups=None, data_format=None,
                 activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros',
                 dilation_rate=(1, 1, 1), depthwise_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, depthwise_constraint=None, bias_constraint=None, **kwargs):
        super(DepthWiseConv3D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.groups = groups
        self.depthwise_initializer = tf.keras.initializers.get(depthwise_initializer)
        self.depthwise_regularizer = tf.keras.regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = _preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        self._data_format = "NDHWC"
        self.input_dim = None

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv3D` '
                             'should be defined. Found `None`.')
        self.input_dim = int(input_shape[channel_axis])

        if self.groups is None:
            self.groups = self.input_dim

        if self.groups > self.input_dim:
            raise ValueError('The number of groups cannot exceed the number of channels')

        if self.input_dim % self.groups != 0:
            raise ValueError('Warning! The channels dimension is not divisible by the group size chosen')

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2],
                                  self.input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.groups * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs):

        if self.data_format == 'channels_last':
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            dilation = self.dilation_rate + (1,) + (1,)

        if self._data_format == 'NCDHW':
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, i:i + self.input_dim // self.groups, :, :, :],
                              self.depthwise_kernel[:, :, :, i:i + self.input_dim // self.groups, :],
                              strides=self._strides,
                              padding=self._padding,
                              dilations=dilation,
                              data_format=self._data_format) for i in
                 range(0, self.input_dim, self.input_dim // self.groups)], axis=1)

        else:
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, :, :, :, i:i + self.input_dim // self.groups],
                              self.depthwise_kernel[:, :, :, i:i + self.input_dim // self.groups, :],
                              strides=self._strides,
                              padding=self._padding,
                              dilations=dilation,
                              data_format=self._data_format) for i in
                 range(0, self.input_dim, self.input_dim // self.groups)], axis=-1)

        if self.bias is not None:
            outputs = tf.keras.backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
            out_filters = self.groups * self.depth_multiplier
        elif self.data_format == 'channels_last':
            depth = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = self.groups * self.depth_multiplier

        depth = conv_output_length(depth, self.kernel_size[0],
                                   self.padding,
                                   self.strides[0])

        rows = conv_output_length(rows, self.kernel_size[1],
                                  self.padding,
                                  self.strides[1])

        cols = conv_output_length(cols, self.kernel_size[2],
                                  self.padding,
                                  self.strides[2])

        if self.data_format == 'channels_first':
            return input_shape[0], out_filters, depth, rows, cols

        elif self.data_format == 'channels_last':
            return input_shape[0], depth, rows, cols, out_filters

    def get_config(self):
        config = super(DepthWiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = tf.keras.initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = tf.keras.regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = tf.keras.constraints.serialize(self.depthwise_constraint)
        return config


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape(axis) if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', channel_order="last_channel"):
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 4 if channel_order == "last_channel" else 1

    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = tf.keras.layers.Conv3D(filters=filters,
                                   kernel_size=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                                   name=prefix + "expand_conv")(inputs)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + "expand_bn")(x)
        x = tf.keras.layers.Activation(activation, name=prefix + "expand_activation")(x)
    else:
        x = inputs

    x = DepthWiseConv3D(kernel_size=block_args.kernel_size,
                        strides=block_args.strides,
                        padding='same',
                        use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        name=prefix + 'dwconv')(x)

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + "bn")(x)
    x = tf.keras.layers.Activation(activation, name=prefix + "activation")(x)

    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = tf.keras.layers.GlobalAveragePooling3D(name=prefix + 'se_squeeze')(x)
        target_shape = (1, 1, 1, filters) if channel_order == 'channels_last' else (filters, 1, 1, 1)
        se_tensor = tf.keras.layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = tf.keras.layers.Conv3D(filters=num_reduced_filters,
                                           kernel_size=1,
                                           activation=activation,
                                           padding="same",
                                           use_bias=True,
                                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                                           name=prefix + "se_reduce")(se_tensor)

        x = tf.keras.layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    x = tf.keras.layers.Conv3D(filters=block_args.output_filters,
                               kernel_size=1,
                               padding="same",
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + "project_conv")(x)

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + "project_bn")(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = FixedDropout(drop_rate, noise_shape=(None, 1, 1, 1, 1), name=prefix + 'drop')(x)
        x = tf.keras.layers.add([x, inputs], name=prefix + "add")

    return x


def efficient_net(width_coefficient, depth_coefficient, default_resolution, dropout_rate=.2, drop_connect_rate=.2,
                  depth_divisor=8, blocks_args=BlockArgs, model_name="3D_efficient_net", input_shape=None, classes=1000,
                  channel_order="last_channel",
                  **kwargs):
    bn_axis = 4 if channel_order == "last_channel" else 1

    img_input = tf.keras.Input(shape=input_shape)

    x = img_input
    x = tf.keras.layers.Conv3D(filters=round_filter(32, width_coefficient, depth_divisor),
                               kernel_size=3,
                               strides=(2, 2, 2),
                               padding="same",
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name="stem_conv")(x)

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
    x = tf.keras.layers.Activation(tf.keras.activations.swish, name="stem_activation")(x)

    # build block
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0

        block_args = block_args._replace(
            input_filters=round_filter(block_args.input_filters,
                                       width_coefficient, depth_divisor),
            output_filters=round_filter(block_args.output_filters,
                                        width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient)
        )

        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args, activation=tf.keras.activations.swish, drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1, 1])
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=tf.keras.activations.swish,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    x = tf.keras.layers.Conv3D(filters=round_filter(1280, width_coefficient, depth_divisor),
                               kernel_size=1,
                               padding="same",
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name="top_conv")(x)

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name="top_bn")(x)
    x = tf.keras.layers.Activation(tf.keras.activations.swish, name="top_activation")(x)

    x = tf.keras.layers.GlobalAveragePooling3D(name="avg_pool")(x)
    if dropout_rate and dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = tf.keras.layers.Dense(classes,
                              activation='softmax',
                              kernel_initializer=DENSE_KERNEL_INITIALIZER,
                              name='probs')(x)

    inputs = img_input

    model = tf.keras.Model(inputs,x,name=model_name)

    return model
