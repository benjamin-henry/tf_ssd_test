import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, PReLU, ReLU, MaxPool2D, Add, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


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


def invResBlock(x: Tensor, index: int,
    expand: int,
    dw_mult: int,
    squeeze: int,
    strides: int = 1,
    use_bias: bool = False,
    l2_reg=.0005) -> Tensor:

    y = Conv2D(expand, 1, 1,padding='valid',use_bias=use_bias, name=f"block_{index}_expand",kernel_regularizer=l2(l2_reg))(x)
    y = DepthwiseConv2D(3, strides,padding='same',use_bias=use_bias, depth_multiplier=dw_mult, name=f"block_{index}_dw",kernel_regularizer=l2(l2_reg))(y)
    # y=PReLU(alpha_initializer=Constant(value=.25))(y)   
    y = BatchNormalization(momentum=0.99, name=f"block_{index}_bn")(y)
    y = Conv2D(squeeze, 1,1, padding='valid',use_bias=use_bias, name=f"block_{index}_project",kernel_regularizer=l2(l2_reg))(y)
        
    if strides==2:
        x = MaxPool2D(2,2, name=f"block_{index}_maxpool2d")(x)
    
    x = pad_depth(x, squeeze, index)
    y = Add(name=f"block_{index}_add")([y, x])
    y = ReLU(name=f"block_{index}_relu")(y)
    return y

