import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Conv2DTranspose, Cropping2D, concatenate,
                                     BatchNormalization)
from tensorflow.keras.models import Model
from unet_models.conv_layers import *

smooth = 1.


# def bn_act(x, act=True):
#     x = keras.layers.BatchNormalization()(x)
#     if act == True:
#         x = keras.layers.Activation("relu")(x)
#     return x
#
#
# def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     conv = bn_act(x)
#     conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
#     return conv
#
#
# def stem(inbound_layer, filters, kernel_size=(3, 3), padding="same", strides=1):
#     #conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(inbound_layer)
#     conv = conv_block(inbound_layer, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#
#     shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(inbound_layer)
#     shortcut = bn_act(shortcut, act=False)
#
#     output = keras.layers.Add()([conv, shortcut])
#     return output
#
#
# def residual_block(inbound_layer, filters, kernel_size=(3, 3), padding="same", strides=1):
#     res = conv_block(inbound_layer, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#     res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
#
#     shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(inbound_layer)
#     shortcut = bn_act(shortcut, act=False)
#
#     output = keras.layers.Add()([shortcut, res])
#     return output
#
#
# def upsample_concat_block(x, xskip):
#     u = keras.layers.UpSampling2D((2, 2))(x)
#     c = keras.layers.Concatenate()([u, xskip])
#     return c


def get_residual_unet_model(input_shape,
                            num_classes=1,
                            dropout=0.5,
                            filters=64,
                            num_layers=4,
                            output_activation='sigmoid'
                            ):
     # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for n in range(num_layers):
        x = res_conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same', num_blocks=1)
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters*2 # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = res_conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same')

    # The next step is to write code for the decoder. In this part, we will go from input of 16x16xK all the way up to 256 x 256 x 1 as output.
    # In the decoder basically we need to upsample and concatenate from corresponding arm of encoder
    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer
        #filters reduces the channel dim and strides reduces the spatial dim
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        # ch, cw = get_crop_shape(int_shape(conv), int_shape(x))
        # conv = Cropping2D(cropping=(ch, cw))(conv)
        x = concatenate([x, conv])
        x = res_conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same')

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# def get_res_unet_model():
#     f = [16, 32, 64, 128, 256]
#     inputs = keras.layers.Input((128, 128, 3))
#     import pdb
#     pdb.set_trace()
#     # Encoder
#     e0 = inputs
#     e1 = stem(e0, f[0])
#     e2 = residual_block(e1, f[1], strides=2)
#     e3 = residual_block(e2, f[2], strides=2)
#     e4 = residual_block(e3, f[3], strides=2)
#     e5 = residual_block(e4, f[4], strides=2)
#
#     # Bridge
#     b0 = conv_block(e5, f[4], strides=1)
#     b1 = conv_block(b0, f[4], strides=1)
#
#     # Decoder
#     u1 = upsample_concat_block(b1, e4)
#     d1 = residual_block(u1, f[4])
#
#     u2 = upsample_concat_block(d1, e3)
#     d2 = residual_block(u2, f[3])
#
#     u3 = upsample_concat_block(d2, e2)
#     d3 = residual_block(u3, f[2])
#
#     u4 = upsample_concat_block(d3, e1)
#     d4 = residual_block(u4, f[1])
#
#     outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
#     pdb.set_trace()
#     model = keras.unet_models.Model(inputs, outputs)
#     return model
#
#
# def get_res_unet_model_w_weights():
#     model_with_all_weights = get_res_unet_model()
#     model_with_all_weights.load_weights("Hep2-Segmentation/unet_models/ResUNet.h5")
#
#     resnet_model = get_res_unet_model()
#
#     for i in range(43):
#         resnet_model.layers[i].set_weights(
#             model_with_all_weights.layers[i].get_weights())
#
#     return resnet_model

# model = get_residual_unet_model((256,256, 1),
#                                 num_classes=1,
#                                 dropout=0.5,
#                                 filters=64,
#                                 num_layers=4,
#                                 output_activation='sigmoid'
#                                 )