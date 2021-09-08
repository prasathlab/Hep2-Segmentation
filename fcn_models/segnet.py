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


def segnet_decoder(f, n_classes, n_up=3):

    assert n_up >= 2
    o = f
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    for _ in range(n_up-2):

        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
             data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)


    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)


    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    if n_classes == 1:
        o = Activation('sigmoid')(o)
    else:
        o = Softmax(axis=-1)(o)

    return o

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

def _segnet(n_classes, encoder,  input_height=416, input_width=608,
            encoder_level=3, channels=3, upsample_level=4):

    assert (n_classes > 1)
    if n_classes == 2:
        n_classes = 1

    img_input, levels = encoder(input_height=input_height,  input_width=input_width, channels=channels)

    feat = levels[encoder_level]
    final_output = segnet_decoder(feat, n_classes, n_up=upsample_level)
    model = Model(img_input, final_output)
    #model = get_segmentation_model(img_input, o)

    return model


def segnet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3, upsample_level=4):
    model = _segnet(n_classes,
                    vanilla_encoder,
                    input_height=input_height,
                    input_width=input_width,
                    encoder_level=encoder_level,
                    channels=channels,
                    upsample_level=upsample_level
                    )
    model.model_name = "segnet"

    return model


def vgg_segnet(n_classes, input_height=416, input_width=608,
               encoder_level=3, channels=3, upsample_level=4,
               pretrained_w=False, fine_tune=True
               ):

    model = _segnet(n_classes,
                    get_vgg_encoder,
                    input_height=input_height,
                    input_width=input_width,
                    encoder_level=encoder_level,
                    channels=channels,
                    upsample_level=upsample_level)
    model.model_name = "vgg_segnet"

    if IMAGE_ORDERING == 'channels_first':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.1/" \
                         "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.1/" \
                         "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if pretrained_w:

        initial_weights = [layer.get_weights() for layer in model.layers]
        weights_path = tf.keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    if not fine_tune:
        model = set_trainable(model, initial_weights)
    return model


def resnet50_segnet(n_classes, input_height=416, input_width=608,
                    encoder_level=3, channels=3, upsample_level=4,
                    pretrained_w=False, fine_tune=True
                    ):

    model = _segnet(n_classes,
                    get_resnet50_encoder,
                    input_height=input_height,
                    input_width=input_width,
                    encoder_level=encoder_level,
                    channels=channels,
                    upsample_level=upsample_level
                    )
    model.model_name = "resnet50_segnet"

    if IMAGE_ORDERING == 'channels_first':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.2/" \
                         "resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.2/" \
                         "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if pretrained_w:

        initial_weights = [layer.get_weights() for layer in model.layers]
        weights_path = tf.keras.utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    if not fine_tune:
        model = set_trainable(model, initial_weights)
    return model


def mobilenet_segnet(n_classes, input_height=224, input_width=224,
                     encoder_level=3, channels=3, upsample_level=4,
                     pretrained_w=False, fine_tune=True):

    model = _segnet(n_classes,
                    get_mobilenet_encoder,
                    input_height=input_height,
                    input_width=input_width,
                    encoder_level=encoder_level,
                    channels=channels,
                    upsample_level=upsample_level)
    model.model_name = "mobilenet_segnet"
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')

    if pretrained_w:
        initial_weights = [layer.get_weights() for layer in model.layers]
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = tf.keras.utils.get_file(model_name, weight_path)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)
    return model


#if __name__ == '__main__':

    #model = segnet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1, upsample_level=4)
    # model = vgg_segnet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1, upsample_level=4,
    #                    pretrained_w=True, fine_tune=False)
    # model = resnet50_segnet(n_classes=2, input_height=256, input_width=256, encoder_level=3, channels=1, upsample_level=4,
    #                    pretrained_w=True, fine_tune=False)

    ##model = mobilenet_segnet(n_classes=3, input_height=256, input_width=256, channels=1, encoder_level=3, upsample_level=4,
     #                  pretrained_w=True, fine_tune=False)

    #debug = 1
    #X = np.random.random(size=(10, 256, 256, 1))

    #m = vgg_segnet(101)
    #m = segnet(101)
    # m = mobilenet_segnet( 101 )
    # from keras.utils import plot_model
    # plot_model( m , show_shapes=True , to_file='model.png')
