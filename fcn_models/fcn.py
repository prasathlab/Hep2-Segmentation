import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder

#from fcn_models.config import IMAGE_ORDERING
#from fcn_models.basic_models import vanilla_encoder
#from fcn_models.vgg16 import get_vgg_encoder
#from fcn_models.model_utils import get_segmentation_model
#from fcn_models.resnet50 import get_resnet50_encoder
#from fcn_models.mobilenet import get_mobilenet_encoder

# from config import IMAGE_ORDERING
# from basic_models import vanilla_encoder
# from vgg16 import get_vgg_encoder
# from resnet50 import get_resnet50_encoder
# from mobilenet import get_mobilenet_encoder

import pdb
# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height2 = o_shape2[2]
        output_width2 = o_shape2[3]
    else:
        output_height2 = o_shape2[1]
        output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    if IMAGE_ORDERING == 'channels_first':
        output_height1 = o_shape1[2]
        output_width1 = o_shape1[3]
    else:
        output_height1 = o_shape1[1]
        output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format=IMAGE_ORDERING)(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format=IMAGE_ORDERING)(o2)

    return o1, o2

def crop_output(o, desired_o ):
    o_shape = o.shape
    if IMAGE_ORDERING == 'channels_first':
        output_height1 = o_shape[2]
        output_width1 = o_shape[3]
    else:
        output_height1 = o_shape[1]
        output_width1 = o_shape[2]
    if IMAGE_ORDERING == 'channels_first':
        desired_height = desired_o.shape[2]
        desired_width = desired_o.shape[3]
    else:
        desired_height = desired_o.shape[1]
        desired_width = desired_o.shape[2]

    assert (output_height1 > desired_height)
    assert (output_width1 > desired_width)
    cx = int(abs(output_width1 - desired_width)/2)
    cy = int(abs(output_height1 - desired_height)/2)

    o1 = Cropping2D(cropping=(cy, cx), data_format=IMAGE_ORDERING)(o)
    return o1


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

def fcn_8(n_classes, encoder=vanilla_encoder, input_height=416, input_width=608, channels=3):

    assert(n_classes>1)
    if n_classes == 2:
        n_classes = 1


    img_input, levels = encoder(input_height=input_height,
                                input_width=input_width,
                                channels=channels,
                                )
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)

    o2 = f4
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                 data_format=IMAGE_ORDERING))(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])

    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o2 = f3
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                 data_format=IMAGE_ORDERING))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add( name="seg_feats" )([o2, o])

    o = Conv2DTranspose(n_classes, kernel_size=(16, 16),  strides=(
        8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)

    final_output = crop_output(o, img_input)
    final_output = Activation('sigmoid')(final_output)
    model = Model(img_input, final_output)



    #model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_8"
    return model



def fcn_32(n_classes, encoder=vanilla_encoder, input_height=416,
           input_width=608, channels=3, ):
    assert (n_classes > 1)
    if n_classes == 2:
        n_classes = 1

    img_input, levels = encoder(input_height=input_height,
                                input_width=input_width,
                                channels=channels,
                                )
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING , name="seg_feats" ))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(64, 64),  strides=(
        32, 32), use_bias=False,  data_format=IMAGE_ORDERING)(o)

    final_output = o
    final_output = crop_output(o, img_input)
    final_output = Activation('sigmoid')(final_output)
    #model = get_segmentation_model(img_input, o)
    model = Model(img_input, final_output)
    model.model_name = "fcn_32"
    return model


def fcn_8_vgg(n_classes,  input_height=416, input_width=608, channels=3, pretrained_w=False, fine_tune=False):
    model = fcn_8(n_classes,
                  get_vgg_encoder,
                  input_height=input_height,
                  input_width=input_width,
                  channels=channels,
                  )
    model.model_name = "fcn_8_vgg"

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
        weights_path = tf.keras.utils.get_file( pretrained_url.split("/")[-1], pretrained_url)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)
    return model


def fcn_32_vgg(n_classes,  input_height=416, input_width=608, channels=3,
               pretrained_w=False, fine_tune=True):

    model = fcn_32(n_classes,
                   get_vgg_encoder,
                   input_height=input_height,
                   input_width=input_width,
                   channels=channels,
                   )
    model.model_name = "fcn_32_vgg"
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
        weights_path = tf.keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)

    return model


def fcn_8_resnet50(n_classes,  input_height=416, input_width=608, channels=3,  pretrained_w=False,
                   fine_tune=True):

    model = fcn_8(n_classes,
                  get_resnet50_encoder,
                  input_height=input_height,
                  input_width=input_width,
                  channels=channels,
                  )
    model.model_name = "fcn_8_resnet50"
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
        weights_path = tf.keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)

    return model


def fcn_32_resnet50(n_classes,  input_height=416, input_width=608, channels=3,
                    pretrained_w=False, fine_tune=True):
    model = fcn_32(n_classes, get_resnet50_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_resnet50"
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


def fcn_8_mobilenet(n_classes,  input_height=224, input_width=224, channels=3,
                    pretrained_w=False, fine_tune=True):
    model = fcn_8(n_classes, get_mobilenet_encoder, input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_mobilenet"
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.6/')

    if pretrained_w:
        initial_weights = [layer.get_weights() for layer in model.layers]
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = tf.keras.utils.get_file(model_name, weight_path)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)

    return model


def fcn_32_mobilenet(n_classes,  input_height=224, input_width=224, channels=3,
                     pretrained_w=False, fine_tune=True):
    model = fcn_32(n_classes, get_mobilenet_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_mobilenet"
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.6/')

    if pretrained_w:
        initial_weights = [layer.get_weights() for layer in model.layers]
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = tf.keras.utils.get_file(model_name, weight_path)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not fine_tune:
        model = set_trainable(model, initial_weights)

    return model

#
# if __name__ == '__main__':
#     debug = 1

#     #model = fcn_8(n_classes=2, encoder=vanilla_encoder, input_height=256, input_width=256, channels=1)
#     #model = fcn_32(n_classes=2, encoder=vanilla_encoder, input_height=256, input_width=256, channels=1)
#
#     #model = fcn_8_vgg(n_classes=2, input_height=256, input_width=256, channels=1)
#     #model = fcn_32_vgg(n_classes=2, input_height=256, input_width=256, channels=1, pretrained_w=True)
#     #model = fcn_8_resnet50(n_classes=2, input_height=256, input_width=256, channels=1, pretrained_w=True)
#     model = fcn_32_mobilenet(n_classes=2, input_height=256, input_width=256, channels=1, pretrained_w=True, fine_tune=False)

#     debug = 1
# #     m = fcn_8(101)
# #     m = fcn_32(101)
