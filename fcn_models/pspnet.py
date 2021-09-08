
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import tensorflow.keras.backend as K

from .config import IMAGE_ORDERING
#from .model_utils import get_segmentation_model, resize_image
from .vgg16 import get_vgg_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder

#
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

def resize_image(inp, s, data_format):
    try:

        return Lambda(lambda x: K.resize_images(x,
                                                height_factor=s[0],
                                                width_factor=s[1],
                                                data_format=data_format,
                                                interpolation='bilinear'))(inp)

    except Exception as e:
        # if keras is old, then rely on the tf function
        # Sorry theano/cntk users!!!
        assert data_format == 'channels_last'
        assert IMAGE_ORDERING == 'channels_last'

        import tensorflow as tf

        return Lambda(
            lambda x: tf.image.resize_images(
                x, (K.int_shape(x)[1]*s[0], K.int_shape(x)[2]*s[1]))
        )(inp)

def pool_block(feats, pool_factor):
    # do avg pooling by computing the pool_size.
    # different pool_sizes imply different granularity of pooled features.

    print(f"pool_block, feats -> {feats.shape}")

    if IMAGE_ORDERING == 'channels_first':
        h = K.int_shape(feats)[2]
        w = K.int_shape(feats)[3]
    elif IMAGE_ORDERING == 'channels_last':
        h = K.int_shape(feats)[1]
        w = K.int_shape(feats)[2]

    pool_size = strides = [int(np.round(float(h) / pool_factor)),
                           int(np.round(float(w) / pool_factor))
                           ]

    print(f"pool size = {pool_size}, pool_factor={pool_factor}")
    x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING,
                         strides=strides, padding='same')(feats)
    print(f"After avg pool, {x.shape}")

    x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING,
               padding='same', use_bias=False)(x)
    print(f"After conv, {x.shape}")

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resize_image(x, strides, data_format=IMAGE_ORDERING)
    print(f"After resize, {x.shape}")

    return x


def _pspnet(n_classes, encoder,  input_height=384, input_width=576, channels=3):

    assert (n_classes > 1)
    if n_classes == 2:
        n_classes = 1
    #assert input_height % 192 == 0
    #assert input_width % 192 == 0

    img_input, levels = encoder(input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5
    pool_factors = [1, 2, 4, 6]
    pool_outs = [o]

    for p in pool_factors:
        #first pool_size = [12, 18]. input size = (None, 12, 18, 256)
            #implies we are doing avg pooling of whole feature map.
            # output size is (None, 12, 18, 512)
        #second pool_size =[6, 9]. input -> (None, 12, 18, 512).
            #this is the input from previous layer
            ##implies we are doing avg pooling of 6x9 blocks on input spatial dims.
            #Note strides is same as pool_size. Implying no overlaps when avg pooling
            #Output shape of (None, 12, 18, 512) is maintained.
        #Subsequent pool blocks just take smaller pool_sizes.
        # implying pool_outs basically contains feature maps at different granularities.

        pooled = pool_block(o, p)
        pool_outs.append(pooled)


    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False , name="seg_feats")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), data_format=IMAGE_ORDERING,
               padding='same')(o)
    o = resize_image(o, (32, 32), data_format=IMAGE_ORDERING)


    if n_classes == 1:
        o = Activation('sigmoid')(o)
    else:
        o = Softmax(axis=-1)(o)

    model = Model(img_input, o)

    return model


def pspnet(n_classes,  input_height=384, input_width=576, channels=3):

    model = _pspnet(n_classes, vanilla_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "pspnet"

    return model


def vgg_pspnet(n_classes,  input_height=384, input_width=576, channels=3, pretrained_w=False, fine_tune=True):

    model = _pspnet(n_classes, get_vgg_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_pspnet"

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


def resnet50_pspnet(n_classes,  input_height=384, input_width=576, channels=3, pretrained_w=False, fine_tune=True):

    model = _pspnet(n_classes, get_resnet50_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_pspnet"

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


def pspnet_50(n_classes,  input_height=473, input_width=473, channels=3):
    from ._pspnet_2_NOT_USED import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 50
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape, channels=channels)
    model.model_name = "pspnet_50"
    return model


def pspnet_101(n_classes,  input_height=473, input_width=473, channels=3):
    from ._pspnet_2_NOT_USED import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 101
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape, channels=channels)
    model.model_name = "pspnet_101"
    return model


# def mobilenet_pspnet( n_classes ,  input_height=224, input_width=224 ):

# 	model =  _pspnet(n_classes, get_mobilenet_encoder,
#                    input_height=input_height, input_width=input_width)
# 	model.model_name = "mobilenet_pspnet"
# 	return model


# if __name__ == '__main__':
#
#     model = pspnet(n_classes=2,   input_height=256, input_width=256, channels=1)
#
#     model = vgg_pspnet(n_classes=2, input_height=256, input_width=256, channels=1, pretrained_w=True, fine_tune=False)
#
#     model = resnet50_pspnet(n_classes=2, input_height=256, input_width=256, channels=1, pretrained_w=True, fine_tune=False)
#
#     #m = _pspnet(101, vanilla_encoder)
#     # m = _pspnet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
#     #m = _pspnet(101, get_vgg_encoder)
#     #m = _pspnet(101, get_resnet50_encoder)
