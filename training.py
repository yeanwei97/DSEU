import os
import numpy as np
import shutil
import glob
import cv2
import matplotlib.pyplot as plt
import glob

import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import ModelCheckpoint

# load dataset
# change the path name accordingly
train_hazy_path = glob.glob('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/OTS_haze/*')
# train_hazy_path = glob.glob('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/ITS_haze/*') 
# train_hazy_path = glob.glob('/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/combine/*')

# set ground truth folder path
gt_path = '/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/clear/'

# set path to save the model weights while training
savepath ="/mnt/956cc712-57b2-4a24-b574-3a113e957774/yw/outdoorunet-{epoch:02d}-{loss:.4f}.hdf5"

# set batch size
batch_size = 6

# cut an image uniformly into 9 256x256 patches
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

# get images in batch
def get_images(path):
  imgs = []
  for i in path:
    img = cv2.imread(i)/255.
    I = img

    # can choose to resize the image or cut it into patches
    # imgs = cut_patches(I) 
    l = cv2.resize(I, (256, 256))

    imgs.append(l)

  return imgs

# get its respective ground truth image in different folder
def get_gt_images(path):
  imgs = []
  for i in path:
    name = i.split('/')[-1]
    name2= name.split('_')[0]

    # ITS dataset has png images while OTS dataset has jpg images
    # this step is to get the right ground truth format by image format
    if i.split('.')[-1] == 'png':
    	img = cv2.imread(gt_path + name2 + '.png')
    else:
    	img = cv2.imread(gt_path + name2 + '.jpg')

    I = img/255.

    # can choose to resize/ cut into patches
    # imgs = cut_patches(I)
    l = cv2.resize(I, (256, 256))
    imgs.append(l)

  return imgs

# load images as batches for training
def train_generator(train_path, batch_size):
  L = len(train_path)
  while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            # X_load as input images
            X_load = get_images(train_path[batch_start:limit])
            # Y_load as ground truth images
            Y_load = get_gt_images(train_path[batch_start:limit])

            # stack the images load as batch
            X = np.stack(X_load, axis=0)
            Y = np.stack(Y_load, axis=0)

            batch_start += batch_size
            batch_end += batch_size

            yield(X, Y)

# for validation purposes if you split the dataset into train/val 
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

# perceptual loss using VGG19
def lossVGG(y_true, y_pred):
	vgg = VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False
	# choose to compare features in 'block2_conv2'
	content_layers = 'block2_conv2'

	lossModel = Model(vgg.input, vgg.get_layer(content_layers).output)

	# load features using output image and ground truth image 
	vggX = lossModel(y_pred)
	vggY = lossModel(y_true)

	return K.mean(K.square(vggX - vggY))

# total loss
def my_loss(y_true, y_pred):
	mse = MSE(y_true, y_pred)
	return lossVGG(y_true, y_pred) + mse

# SE block
def squeeze_excite_block(tensor):
	init = tensor
	channel_axis = 3
	filters = init.shape[channel_axis]
	se_shape = (1, 1, filters)

	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	# set ratio to reduce the complexity of output channel as 8
	r = 8
	se = Dense(filters // r, activation='relu', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid', use_bias=False)(se)

	x = multiply([init, se])
	return x

# model implementation
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

# load model
model = dseu()

# show layers in model
print(model.summary())

# save model for each iteration
checkpoint = ModelCheckpoint(savepath,
                             monitor='loss',
                             save_best_only=False
                             )
callbacks_list = [checkpoint]

# training
model.fit_generator(train_generator(train_hazy_path, batch_size),
                    steps_per_epoch=20000,
                    epochs = 20,
                    callbacks=callbacks_list,
                    verbose=1)
K.clear_session()
