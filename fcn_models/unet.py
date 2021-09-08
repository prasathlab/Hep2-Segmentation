import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np

from .config import IMAGE_ORDERING
#from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder

# from config import IMAGE_ORDERING
# from basic_models import vanilla_encoder
# from vgg16 import get_vgg_encoder
# from resnet50 import get_resnet50_encoder
# from mobilenet import get_mobilenet_encoder

import pdb

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def unet_mini(n_classes, input_height=360, input_width=480, channels=3):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)

    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)

    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same' , name="seg_feats")(conv5)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv5)

    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_mini"
    return model


def set_trainable(model, initial_weights):
    for idx, (layer, initial) in enumerate(zip(model.layers, initial_weights)):
        weights = layer.get_weights()
        if weights:
            print(f"{layer.name}, initial = {initial[0].shape}, weights= {weights[0].shape}")
            if not all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'Checkpoint contained weights for layer {layer.name}!')
                layer.trainable = False
            else:
                print(f'Checkpoint contained No weights for layer {layer.name}!')
    return model


def unet_upsampling_decoder(levels, n_classes):
    [f1, f2, f3, f4, f5] = levels
    o = f4
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)


    o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)  # name="seg_feats"
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

    if n_classes == 1:
        o = Activation('sigmoid')(o)
    else:
        o = Softmax(axis=-1)(o)
    return o


def unet_transconv_decoder(levels, n_classes):
    [f1, f2, f3, f4, f5] = levels

    o = f5
    print(f"Initial: {o.shape}")
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(1024, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    filters = f4.shape[MERGE_AXIS]
    o = (Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2),data_format=IMAGE_ORDERING))(o)
    print(f"Before Concat: {o.shape}, {f4.shape}")
    o = (concatenate([o, f4], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    filters = f3.shape[MERGE_AXIS]
    o = (Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2),data_format=IMAGE_ORDERING))(o)
    print(f"Before Concat: {o.shape}, {f3.shape}")
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)


    filters = f2.shape[MERGE_AXIS]
    o = (Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), data_format=IMAGE_ORDERING))(o)
    print(f"Before Concat: {o.shape}, {f2.shape}")
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)


    filters = f1.shape[MERGE_AXIS]
    o = (Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), data_format=IMAGE_ORDERING))(o)
    print(f"Before Concat: {o.shape}, {f1.shape}")
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)  # name="seg_feats"
    o = (BatchNormalization())(o)


    print(f"Before Final Upsampling: {o.shape}")
    o = (Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), data_format=IMAGE_ORDERING))(o)
    print(f"After Final Upsampling: {o.shape}")
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

    if n_classes == 1:
        o = Activation('sigmoid')(o)
    else:
        o = Softmax(axis=-1)(o)
    return o

def _unet(n_classes, encoder, input_height=416, input_width=608, channels=3):
    assert (n_classes > 1)
    if n_classes == 2:
        n_classes = 1
    img_input, levels = encoder(input_height=input_height, input_width=input_width, channels=channels)

    final_output = unet_transconv_decoder(levels, n_classes)

    model = Model(img_input, final_output)
    return model


def unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):
    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "unet"
    return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3,
             pretrained_w=False, fine_tune=True):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_unet"

    if IMAGE_ORDERING == 'channels_first':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.1/" \
                         "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.1/" \
                         "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    print("Before")
    model.summary()
    if pretrained_w:
        initial_weights = [layer.get_weights() for layer in model.layers]
        weights_path = tf.keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)
    print("After")
    model.summary()
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3, channels=3, pretrained_w=False, fine_tune=True):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_unet"

    if IMAGE_ORDERING == 'channels_first':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.2/" \
                         "resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.2/" \
                         "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    print("Before")
    model.summary()
    if pretrained_w:
        initial_weights = [layer.get_weights() for layer in model.layers]
        weights_path = tf.keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)

    print("Before")
    model.summary()

    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3, channels=3, pretrained_w=False, fine_tune=True):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "mobilenet_unet"
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')
    print("Before")
    model.summary()
    if pretrained_w:
        initial_weights = [layer.get_weights() for layer in model.layers]
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = tf.keras.utils.get_file(model_name, weight_path)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)

    print("After")
    model.summary()
    return model

#
# if __name__ == '__main__':
#     #m = unet_mini(101)
#
#     model = unet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1)
#
#
#     model = vgg_unet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1,
#                      pretrained_w=True, fine_tune=False)
#
#
#     model = resnet50_unet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1,
#                      pretrained_w=True, fine_tune=False)
#
#
#     model = mobilenet_unet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1,
#                      pretrained_w=True, fine_tune=False)
#
