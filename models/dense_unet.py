import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Conv2DTranspose, Cropping2D, concatenate,
                                     Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from models.conv_layers import *

smooth = 1.


def get_dense_unet_model(input_shape,
                         num_classes=2,
                         dropout=0.5,
                         filters=64,
                         num_layers=4,
                         transition=True,
                         compression_factor=0.5
                         ):
    #First get the correct number of classes
    assert(num_classes > 1)
    if num_classes == 2:
        num_output_channels = 1
        output_activation = 'sigmoid'
    else:
        num_output_channels = num_classes
        output_activation = 'softmax'
     # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for n in range(num_layers):
        x = dense_conv2d_layer(inbound_layer=x,
                               filters=filters,
                               use_batch_norm=True,
                               dropout=dropout,
                               padding='same',
                               num_blocks=1,
                               transition=transition,
                               compression_factor=compression_factor)
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters*2 # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = dense_conv2d_layer(inbound_layer=x,
                           filters=filters,
                           use_batch_norm=True,
                           dropout=dropout,
                           padding='same',
                           num_blocks=2,
                           transition=transition,
                           compression_factor=compression_factor
                           )

    # The next step is to write code for the decoder. In this part, we will go from input of 16x16xK all the way up to 256 x 256 x 1 as output.
    # In the decoder basically we need to upsample and concatenate from corresponding arm of encoder
    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer
        #filters reduces the channel dim and strides reduces the spatial dim
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=-1)([x, conv])
        x = conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same')

    outputs = Conv2D(filters=num_output_channels, kernel_size=(1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# debug = 1
# model = get_dense_unet_model((256,256, 1),
#                                 num_classes=2,
#                                 dropout=0.5,
#                                 filters=64,
#                                 num_layers=4,
#                                 transition=True
#                                 )
# debug = 2