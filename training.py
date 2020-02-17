import os
import numpy as np
import shutil
import glob
import cv2
import matplotlib.pyplot as plt

import glob
train_hazy_path = glob.glob('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/OTS_haze/*')
train_hazy_path = glob.glob('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/ITS_haze/*')
train_hazy_path = glob.glob('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/combine/*')

import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.losses import MSE

def cut_patches(img):
    patches = []
    width = 0
    height = 0

    for i in range(2):
    	width = 0
    	if i == 1:
    		height = img.shape[0] - 256

    	for j in range(3):
    		if j == 1:
    			width = (img.shape[1] // 2) - 128
    		elif j == 2:
    			width = img.shape[1] - 256

    		cut = img[height:height+256, width:width+256]
    		patches.append(cut)

    return patches

def get_images(path):
  imgs = []
  for i in path:
    img = cv2.imread(i)/255.
    I = img
    # imgs = cut_patches(I)
    l = cv2.resize(I, (256, 256))
    imgs.append(l)

  return imgs

def get_gt_images(path):
  imgs = []
  for i in path:
    name = i.split('/')[-1]
    name2= name.split('_')[0]
    if i.split('.')[-1] == 'png':
    	img = cv2.imread('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/clear/' + name2 + '.png')
    else:
    	img = cv2.imread('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/clear/' + name2 + '.jpg')
    I = img/255.
    # imgs = cut_patches(I)
    l = cv2.resize(I, (256, 256))
    imgs.append(l)

  return imgs

def train_generator(train_path, batch_size):
  L = len(train_path)
  while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X_load = get_images(train_path[batch_start:limit])
            Y_load = get_gt_images(train_path[batch_start:limit])
            X = np.stack(X_load, axis=0)
            Y = np.stack(Y_load, axis=0)

            batch_start += batch_size
            batch_end += batch_size

            yield(X, Y)

def val_generator(val_path, batch_size):
  L = len(val_path)
  while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X_load = get_images(val_path[batch_start:limit])
            Y_load = get_gt_images(val_path[batch_start:limit])
            X = np.stack(X_load, axis=0)
            Y = np.stack(Y_load, axis=0)

            batch_start += batch_size
            batch_end += batch_size

            yield (X, Y)

def lossVGG(y_true, y_pred):
	vgg = VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False
	content_layers = 'block2_conv2'

	lossModel = Model(vgg.input, vgg.get_layer(content_layers).output)

	vggX = lossModel(y_pred)
	vggY = lossModel(y_true)

	return K.mean(K.square(vggX - vggY))

def my_loss(y_true, y_pred):
	mse = MSE(y_true, y_pred)
	return lossVGG(y_true, y_pred) + mse

def squeeze_excite_block(tensor):
	init = tensor
	channel_axis = 3
	filters = init.shape[channel_axis]
	se_shape = (1, 1, filters)

	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // 8, activation='relu', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid', use_bias=False)(se)

	x = multiply([init, se])
	return x

def dseu():
    inputs = Input((None, None, 3))

    # Encoder part
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    dilate1 = Conv2D(256, (3, 3), padding='same', dilation_rate=1, kernel_initializer='he_normal')(pool4)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)
    dilate2 = Conv2D(256, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal')(pool4)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)
    dilate3 = Conv2D(256, (3, 3), padding='same', dilation_rate=4, kernel_initializer='he_normal')(pool4)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)
    dilate4 = Conv2D(256, (3, 3), padding='same', dilation_rate=8, kernel_initializer='he_normal')(pool4)
    dilate4 = BatchNormalization()(dilate4)
    dilate4 = Activation('relu')(dilate4)
    concat_dilate = concatenate([dilate1, dilate2, dilate3, dilate4])

    # Decoder
    up5 = Conv2D(512, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(concat_dilate))
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    conv4 = squeeze_excite_block(conv4)
    merge5 = concatenate([conv4, up5], axis = 3)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2D(256, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    conv3 = squeeze_excite_block(conv3)
    merge6 = concatenate([conv3, up6], axis = 3)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2D(128, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    conv2 = squeeze_excite_block(conv2)
    merge7 = concatenate([conv2, up7], axis = 3)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2D(64, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    conv1 = squeeze_excite_block(conv1)
    merge8 = concatenate([conv1, up8], axis = 3)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Conv2D(3, (1, 1), activation = 'tanh')(conv8)

    model = Model(inputs, conv9)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss=my_loss, optimizer=adam, metrics=['accuracy'])
    return model

def baselineUnet():
    inputs = Input((None, None, 3))

    # Encoder part
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    middle = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)
    middle = BatchNormalization()(middle)
    middle = Activation('relu')(middle)
    middle = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(middle)
    middle = BatchNormalization()(middle)
    middle = Activation('relu')(middle)

    # Decoder
    up5 = Conv2D(512, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(middle))
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    merge5 = concatenate([conv4, up5], axis = 3)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2D(256, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = concatenate([conv3, up6], axis = 3)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2D(128, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = concatenate([conv2, up7], axis = 3)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2D(64, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    merge8 = concatenate([conv1, up8], axis = 3)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Conv2D(3, (1, 1), activation = 'tanh')(conv8)

    model = Model(inputs, conv9)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss=my_loss, optimizer=adam, metrics=['accuracy'])
    return model

def dilationUnet():
    inputs = Input((None, None, 3))

    # Encoder part
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    dilate1 = Conv2D(256, (3, 3), padding='same', dilation_rate=1, kernel_initializer='he_normal')(pool4)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)
    dilate2 = Conv2D(256, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal')(pool4)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)
    dilate3 = Conv2D(256, (3, 3), padding='same', dilation_rate=4, kernel_initializer='he_normal')(pool4)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)
    dilate4 = Conv2D(256, (3, 3), padding='same', dilation_rate=8, kernel_initializer='he_normal')(pool4)
    dilate4 = BatchNormalization()(dilate4)
    dilate4 = Activation('relu')(dilate4)
    concat_dilate = concatenate([dilate1, dilate2, dilate3, dilate4])

    # Decoder
    up5 = Conv2D(512, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(concat_dilate))
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    merge5 = concatenate([conv4, up5], axis = 3)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2D(256, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = concatenate([conv3, up6], axis = 3)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2D(128, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = concatenate([conv2, up7], axis = 3)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2D(64, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    merge8 = concatenate([conv1, up8], axis = 3)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Conv2D(3, (1, 1), activation = 'tanh')(conv8)

    model = Model(inputs, conv9)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss=my_loss, optimizer=adam, metrics=['accuracy'])
    return model

def baselineSEUnet():
    inputs = Input((None, None, 3))

    # Encoder part
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    middle = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)
    middle = BatchNormalization()(middle)
    middle = Activation('relu')(middle)
    middle = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(middle)
    middle = BatchNormalization()(middle)
    middle = Activation('relu')(middle)

   # Decoder
    up5 = Conv2D(512, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(middle))
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    conv4 = squeeze_excite_block(conv4)
    merge5 = concatenate([conv4, up5], axis = 3)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2D(256, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    conv3 = squeeze_excite_block(conv3)
    merge6 = concatenate([conv3, up6], axis = 3)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2D(128, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    conv2 = squeeze_excite_block(conv2)
    merge7 = concatenate([conv2, up7], axis = 3)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2D(64, (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    conv1 = squeeze_excite_block(conv1)
    merge8 = concatenate([conv1, up8], axis = 3)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Conv2D(3, (1, 1), activation = 'tanh')(conv8)

    model = Model(inputs, conv9)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss=my_loss, optimizer=adam, metrics=['accuracy'])
    return model


model = dseu()
print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint

filepath="/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/outdoorunet-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             save_best_only=False
                             )
callbacks_list = [checkpoint]

model.fit_generator(train_generator(train_hazy_path, 6),
                    steps_per_epoch=20000,
                    epochs = 20,
                    callbacks=callbacks_list,
                    verbose=1)
K.clear_session()
