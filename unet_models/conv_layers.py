
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Conv2DTranspose, Cropping2D, concatenate,
                                     BatchNormalization, Add, Concatenate)

from tensorflow.keras.backend import int_shape


# Simple 2D conv block.
# We might want to try other blocks ex: Residual Blocks, Inception blocks, Dense connection blocks etc.
def conv2d_layer(inbound_layer,
                 use_batch_norm=True,
                 dropout=0.5,
                 dropout_type="spatial",
                 filters=16,
                 kernel_size=(3, 3),
                 activation="relu",
                 kernel_initializer="he_normal",
                 padding="same",
                 num_blocks=1
                 ):    
    for i in range(num_blocks):
        inbound_layer = Conv2D(filters,
               kernel_size,
               activation=activation,
               kernel_initializer=kernel_initializer,
               padding=padding,
               use_bias=not use_batch_norm,)(inbound_layer)
    if use_batch_norm:
        inbound_layer = BatchNormalization()(inbound_layer)
    if dropout > 0.0:
        inbound_layer = Dropout(dropout)(inbound_layer)

    return inbound_layer

def res_conv2d_layer(inbound_layer,
                     use_batch_norm=True,
                     dropout=0.5,
                     dropout_type="spatial",
                     filters=16,
                     kernel_size=(3, 3),
                     activation="relu",
                     kernel_initializer="he_normal",
                     padding="same",
                     num_blocks=1
                     ):
    identity = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=1)(inbound_layer)
    for i in range(num_blocks):
        inbound_layer = Conv2D(filters,
                               kernel_size,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               padding=padding,
                               use_bias=not use_batch_norm,)(inbound_layer)
    if use_batch_norm:
        inbound_layer = BatchNormalization()(inbound_layer)
    if dropout > 0.0:
        inbound_layer = Dropout(dropout)(inbound_layer)

    outbound_layer = Add()([inbound_layer, identity])
    return outbound_layer

def dense_conv2d_layer(inbound_layer,
                       use_batch_norm=True,
                       dropout=0.5,
                       dropout_type="spatial",
                       filters=16,
                       kernel_size=(3, 3),
                       activation="relu",
                       kernel_initializer="he_normal",
                       padding="same",
                       transition=False,
                       compression_factor=0.5,
                       num_blocks=1
                       ):
    #print(f"First inbound: {inbound_layer.shape}")
    for i in range(num_blocks):
        #print(f"block {i}")
        conv_layer = Conv2D(filters,
                            kernel_size,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            padding=padding,
                            use_bias=not use_batch_norm,
                            )(inbound_layer)

        inbound_layer = Concatenate(axis=-1)([inbound_layer, conv_layer])
        #print(f"conv: {conv_layer.shape}, concat: {inbound_layer.shape}")

        if use_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)
        if dropout > 0.0:
            inbound_layer = Dropout(dropout)(inbound_layer)
    if transition:
        n_filters = round(inbound_layer.shape[-1]*compression_factor)
        inbound_layer = Conv2D(n_filters,
                               kernel_size,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               padding=padding,
                               use_bias=not use_batch_norm,
                               )(inbound_layer)
        if use_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)
        if dropout > 0.0:
            inbound_layer = Dropout(dropout)(inbound_layer)
        #print(f"After Transition : {inbound_layer.shape}")
    return inbound_layer