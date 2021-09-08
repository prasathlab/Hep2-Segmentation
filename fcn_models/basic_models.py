from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

#from fcn_models.config import IMAGE_ORDERING
from .config import IMAGE_ORDERING

def vanilla_encoder(input_height=224,  input_width=224, channels=3):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    x = img_input
    print(f"Input shape: {x.shape}")
    levels = []
    print("first layer")
    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    print(f"After Zero Pad: {x.shape}")
    x = (Conv2D(filter_size, (kernel, kernel),
                data_format=IMAGE_ORDERING, padding='valid'))(x)
    print(f"After Conv shape: {x.shape}")
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    print(f"After Max Pool shape: {x.shape}")
    levels.append(x)

    print("Second layer")
    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    print(f"After Zero Pad: {x.shape}")
    x = (Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
         padding='valid'))(x)
    print(f"After Conv shape: {x.shape}")
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    print(f"After Max Pool shape: {x.shape}")
    levels.append(x)

    for idx in range(3):
        print(f"Layer idx = {idx + 2}")
        
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        print(f"After Zero Pad: {x.shape}")
        x = (Conv2D(256, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        print(f"After Conv shape: {x.shape}")
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        print(f"After Max Pool shape: {x.shape}")

        levels.append(x)
    return img_input, levels
