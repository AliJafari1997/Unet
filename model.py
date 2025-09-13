import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input, AveragePooling2D
from tensorflow.keras.models import Model
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_model(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    par1 = partial_mesh(s1, s2, s3, s4, 4, 64, strides = 1, ratio=8, kernel_size=7)
    par2 = partial_mesh(s1, s2, s3, s4, 3, 128, strides = 1, ratio=8, kernel_size=7)
    par3 = partial_mesh(s1, s2, s3, s4, 2, 256, strides = 1, ratio=8, kernel_size=7)
    par4 = partial_mesh(s1, s2, s3, s4, 1, 512, strides = 1, ratio=8, kernel_size=7)

    d1 = decoder_block(b1, par4, 512)
    d2 = decoder_block(d1, par3, 256)
    d3 = decoder_block(d2, par2, 128)
    d4 = decoder_block(d3, par1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_model(input_shape)
    model.summary()


