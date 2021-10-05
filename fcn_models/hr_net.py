import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import tensorflow.keras.backend as K

from .config import IMAGE_ORDERING
#from config import IMAGE_ORDERING


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, kernel_size=(3,3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, kernel_size=(1,1), strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def stem_net(input):
    #Do one initial 3x3 convolution.
    x = Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Each bottleneck_block is just a Residual Block
    x = bottleneck_Block(x, 256, with_conv_shortcut=True)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)

    return x


def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    #x1 is result of downsampling x
    x1 = Conv2D(out_filters_list[1], kernel_size=(3, 3), strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, kernel_size=(3,3), padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, kernel_size=(1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def make_branch(x, out_filters=32):
    #Essentially: just convolutions on the 1x branch, No downsampling etc.
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer1(x):
    #Fuse 1x and 2x layers.
    #Part 1: Upsample the 2x layer and add it to 1x layer
    x0_0 = x[0]
    # Do conv on 2x to make sure 1x and 2x have same number of filter maps
    x0_1 = Conv2D(32, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    #This is the crucial upsampling bit
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    #Add them: this is the new 1x layer
    x0 = add([x0_0, x0_1])

    #Part 2: Downsample the 1x layer and add it to the 2x layer
    #Do conv on x[0] with stride 2 to downsample the 1x layer
    x1_0 = Conv2D(64, kernel_size=(3,3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    # Add them: this is the new 2x layer
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def fuse_layer2(x):
    #x[0] -> 1x
    # x[1] -> 2x
    # x[2] -> 4x
    #Part 1: Upsample the 2x and 4x layers and add it to 1x layer
    x0_0 = x[0]
    # Do conv on 2x to make sure 1x and 2x have same number of filter maps
    x0_1 = Conv2D(32, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    #This is the crucial upsampling bit
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)
    # Do conv on 4x to make sure 1x, 2x and 4x have same number of filter maps
    x0_2 = Conv2D(32, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    # This is the crucial upsampling bit
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)
    #Add them: this is the new 1x layer
    x0 = add([x0_0, x0_1, x0_2])

    #Part 2: Downsample the 1x layer to 2x and upsample 4x layer to 2x, then add both of these to 2x layer
    x1_0 = Conv2D(64, kernel_size=(3,3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(64, kernel_size=(1, 1), use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    # This is the crucial upsampling bit
    x1_2 = UpSampling2D(size=(2, 2))(x1_2)
    # Add them: this is the new 2x layer
    x1 = add([x1_0, x1_1, x1_2])

    # Part 3: Downsample the 1x layer to 4x, down 2x layer to 4x, then add both of these to 4x layer
    x2_0 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                  kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                  kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                  kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_2 = x[2]
    # Add them: this is the new 2x layer
    x2 = add([x2_0, x2_1, x2_2])

    return [x0, x1, x2]

def transition_layer2(x, out_filters_list=[32, 64, 128]):
    # x[0] --> this is the 1x branch
    # x[1] --> this is the 2x branch

    # this transition layer is essentially to get a new 4x branch.
    # 4x branch is obtained by downsampling the 2x (x[1]) branch.
    # Everything else is just some more convs (because ...why not).

    x0 = Conv2D(out_filters_list[0], kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    # this is the main downsampling of 2x layer to create the 4x layer
    x2 = Conv2D(out_filters_list[2], kernel_size=(3,3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]

def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    #Timepass
    x0 = Conv2D(out_filters_list[0], kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    # Timepass
    x1 = Conv2D(out_filters_list[1], kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    # Timepass
    x2 = Conv2D(out_filters_list[2], kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    # Downsample x[2] to get x3 -- this is the new 8x downsample layer
    x3 = Conv2D(out_filters_list[3], kernel_size=(3,3), strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]

def fuse_layer3(x):
    #x[0] -> 1x, x[1] -> 2x, x[2] -> 4x, x[3] -> 8x
    # Part 1: Upsample the 2x, 4x and 8x layers and add it to 1x layer
    x0_0 = x[0]
    # Do conv on 2x to make sure 1x and 2x have same number of filter maps
    x0_1 = Conv2D(32, kernel_size=(1, 1), use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2))(x0_1)

    x0_2 = Conv2D(32, kernel_size=(1, 1), use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4))(x0_2)

    x0_3 = Conv2D(32, kernel_size=(1, 1), use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = BatchNormalization(axis=3)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8))(x0_3)
    # Add them: this is the new 1x layer

    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    #Fusion for other layers is ignored. since, we don't need them
    return x0


def final_layer(x, n_classes=1):
    x = Conv2D(n_classes, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if n_classes == 1:
        x = Activation('sigmoid', name='Pixel_Classification')(x)
    else:
        x = Softmax(axis=-1)(x)
    return x


def hr_net(n_classes, input_height=256, input_width=256, channels=1):
    assert (n_classes > 1)
    if n_classes == 2:
        n_classes = 1
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    x = stem_net(img_input)
    x = transition_layer1(x)
    x0 = make_branch(x[0], out_filters=32)
    x1 = make_branch(x[1], out_filters=64)
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch(x[0], out_filters=32)
    x1 = make_branch(x[1], out_filters=64)
    x2 = make_branch(x[2], out_filters=128)
    x = fuse_layer2([x0, x1, x2])

    x = transition_layer3(x)
    x0 = make_branch(x[0], out_filters=32)
    x1 = make_branch(x[1], out_filters=64)
    x2 = make_branch(x[2], out_filters=128)
    x3 = make_branch(x[3], out_filters=256)

    x = fuse_layer3([x0, x1, x2, x3])
    x = final_layer(x, n_classes=n_classes)

    model = Model(img_input, x)
    model.model_name = "hr_net"
    # import pdb
    # pdb.set_trace()
    return model


#model =  hr_net(n_classes=2, input_height=256, input_width=256, encoder_level=4, channels=1)

#debug = 1