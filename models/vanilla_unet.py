
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Conv2DTranspose, Cropping2D, concatenate,
                                     BatchNormalization)
from tensorflow.keras.models import Model
from models.conv_layers import *

def get_vanilla_unet_model( input_shape,
                num_classes=1,
                dropout=0.5,
                filters=64,
                num_layers=4,
                output_activation='sigmoid'):
    # Unet has 2 parts. First is encoder and secod is decoder
    # Encoder: Basica idea is to reduce the spatial dimension after each block while increasing the channel/fmap dimensions
    # For example:
    # Our image is 256x256. So our first conv block will take 256x256 input and will output 256 X 256 X n_fmaps
    # After this we will use Maxpooling to reduce the spatial dimension by factor of 2. So after maxpooling we should have 128 X 128 X n_fmaps
    # Our next conv block will take 128 X 128 X n_fmaps as input and produce 128 X 128 X n_fmaps*2 as output.
    # After this we will use Maxpooling to reduce the spatial dimension by factor of 2. So after maxpooling we should have 64 X 64 X n_fmaps*2 as output.
    # Technically we can continue this process to get 1x1xK dimension but that is not recommended. Instead we will stop at something like 32 x 32 x K or 16 x 16 x K.
    # This concludes the encoder.

    # Input: 256x256x1 #output: 256x256x1: Every pixel has to be classified.
    # E1: 256x256x64 -> Hopefully would have discovered some low level features
    # Mp1: 128x128x64: (2,2), stride=2

    # E2: 128x128x128 -> Using low level features in E1. Model will now learn slightly higher level features
    # Mp2: 64x64x128

    # E3: 64x64x256
    # Mp3: 32x32x256

    # E4: 32x32x512
    # Mp4: 16x16x512

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same')
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters*2 # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same')

    # The next step is to write code for the decoder. In this part, we will go from input of 16x16xK all the way up to 256 x 256 x 1 as output.
    # In the decoder basically we need to upsample and concatenate from corresponding arm of encoder
    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        # ch, cw = get_crop_shape(int_shape(conv), int_shape(x))
        # conv = Cropping2D(cropping=(ch, cw))(conv)
        x = concatenate([x, conv])
        x = conv2d_layer(inbound_layer=x, filters=filters, use_batch_norm=True, dropout=dropout, padding='same')

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw /2), int(cw /2) + 1
    else:
        cw1, cw2 = int(cw /2), int(cw /2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch /2), int(ch /2) + 1
    else:
        ch1, ch2 = int(ch /2), int(ch /2)

    return (ch1, ch2), (cw1, cw2)