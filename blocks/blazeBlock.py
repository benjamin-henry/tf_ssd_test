"""
from https://github.com/minus31/BlazeFace/
"""

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, PReLU, ReLU, MaxPool2D, Add, BatchNormalization, MaxPool2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def channel_padding(x):
    """
    zero padding in an axis of channel 
    """
    return K.concatenate([x, tf.zeros_like(x)], axis=-1)

def pad_depth(x, desired_channels,index):
    shape = x.shape.as_list()
    assert len(shape) == 4
    new_channels = desired_channels - shape[-1]
    if new_channels > 0:
        y = K.zeros_like(x,name=f"block_{index}_zeros_like")
        y = y[:,:,:,:new_channels]
        return K.concatenate([x, y])
    else:
        return x


def singleBlazeBlock(x: Tensor, index:int, filters:int=24, depth_multiplier:int=1, kernel_size:int=5, strides:int=1, padding:str='same') -> Tensor:

    # depth-wise separable convolution
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        use_bias=False,
        name=f"block_{index}_separable_conv1")(x)

    x_1 = BatchNormalization(name=f"block_{index}_batch_norm")(x_0)

    # Residual connection
    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_1.shape[-1]
        
        x_ = MaxPool2D(name=f"block_{index}_pooling")(x)

        if output_channels - input_channels != 0:
            # channel padding
            x_ = tf.keras.layers.Lambda(channel_padding,name=f"block_{index}_channel_pad")(x_)
        out = Add(name=f"block_{index}_add")([x_1, x_])
        return ReLU(name=f"block_{index}_relu")(out)

    out = Add(name=f"block_{index}_add")([x_1, x])
    return ReLU(name=f"block_{index}_relu")(out)


def doubleBlazeBlock(x: Tensor, index:int, filters_1:int=24, depth_multiplier:int=1, filters_2:int=96, kernel_size:int=5, strides:int=1, padding:str='same') -> Tensor:

    # depth-wise separable convolution
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        depth_multiplier=depth_multiplier,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=f"block_{index}_separable_conv1")(x)

    x_1 = BatchNormalization(name=f"block_{index}_batchnorm1")(x_0)

    x_2 = ReLU(name=f"block_{index}_relu1")(x_1)

    # depth-wise separable convolution, expand
    x_3 = tf.keras.layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        depth_multiplier=depth_multiplier,
        use_bias=False,
        name=f"block_{index}_separable_conv2")(x_2)

    x_4 = BatchNormalization(name=f"block_{index}_batchnorm2")(x_3)

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_4.shape[-1]

        x_ = MaxPool2D(name=f"block_{index}_pooling")(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = Lambda(channel_padding,name=f"block_{index}_channel_pad")(x_)

        out = Add(name=f"block_{index}_add")([x_4, x_])
        return ReLU(name=f"block_{index}_relu")(out)

    out = Add(name=f"block_{index}_add")([x_4, x])
    return ReLU(name=f"block_{index}_relu")(out)