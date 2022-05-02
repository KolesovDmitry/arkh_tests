# -*- coding: utf-8 -*-

import os
import random

from pprint import pprint

import numpy as np

from pathlib import Path 

from tifffile import tifffile as tiff

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import models
from keras import losses
from keras import metrics
from tensorflow.keras.utils import Sequence


# Specify names locations
FOLDER = '/data/Alarm/Samples/2021.05.01'
TRAINING_BASE = 'probas_.tif'
VALID_PIXELS = 'used_pixels.tif'

TRAIN_FILE_PATH = os.path.join(FOLDER, TRAINING_BASE+'.tif')
VALIDATION_RATIO = 0.33 

MODEL_PATH = '/data/Alarm/Samples/Model'

DATA_FILES = list(Path(FOLDER).rglob("probas_.tif"))
random.shuffle(DATA_FILES) 
validation_count = int(len(DATA_FILES) * VALIDATION_RATIO)

TRAIN_DATA = sorted(DATA_FILES[validation_count:])
VALIDATION_DATA = sorted(DATA_FILES[:validation_count])

INPUT_BANDS = ['day_num', 'evi', 'ndvi', 'gndvi', 'nrgi', 'rvi', 'ndwi', 'savi'] 
TARGET_BANDS = ['EVI', 'NDVI', 'GNDVI', 'NRGI', 'RVI', 'NDWI', 'SAVI']
FEATURE_NAMES = INPUT_BANDS + TARGET_BANDS


class TiffReader:
    def __init__(self, input_band_count=8):
        self.input_band_count = input_band_count

    def get_sample_image(self, filename):
        x = tiff.imread(filename)
        x = np.nan_to_num(x)   
        y = x[:, :, self.input_band_count:]
        mask = np.logical_not(np.isnan(y).any(axis=2))
        mask = mask.astype(int)
        
        return (x[:, :, :self.input_band_count], y, mask)


class FileSequence(Sequence):
    def __init__(self, filenames, batch_size, repeats=32, width=1024, input_band_count=8):
        self.filenames = filenames
        random.shuffle(self.filenames)
        self.batch_size = batch_size
        self.width = width
        self.input_band_count = input_band_count
        self.tiff_reader = TiffReader(input_band_count=self.input_band_count)

        self.iter_count = 0
        self.max_iter_count = repeats

    def __len__(self):
        return len(self.filenames) * self.max_iter_count

    def __getitem__(self, idx):
        return self.shrade_ds(idx)

    def shrade_ds(self, idx):
        idx = idx % len(self.filenames)
        filename = self.filenames[idx]
        x, y, mask = self.get_sample_image(filename)
        rows, cols = x.shape[0], x.shape[1]
        i = np.random.randint(low=0, high=rows-self.width, size=self.batch_size)
        j = np.random.randint(low=0, high=cols-self.width, size=self.batch_size)

        xdata = np.array(
            [x[i[k]: i[k]+self.width, j[k]: j[k]+self.width, :] for k in range(self.batch_size)]
        )
        ydata = np.array(
            [y[i[k]: i[k]+self.width, j[k]: j[k]+self.width, :] for k in range(self.batch_size)]
        )
        maskdata = np.array(
            [mask[i[k]: i[k]+self.width, j[k]: j[k]+self.width] for k in range(self.batch_size)]
        )

        return xdata, ydata, maskdata

    def get_sample_image(self, filename):
        return self.tiff_reader.get_sample_image(filename)



def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    # encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    # encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    # encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    # decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    # decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    # decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation('relu')(decoder)
    return decoder


def get_model(input_band_count=8, output_band_count=7, loss='MeanAbsoluteError', show_metrics=['RootMeanSquaredError']):
    inputs = layers.Input(shape=[None, None, input_band_count])
    encoder0_pool, encoder0 = encoder_block(inputs, 64)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    center = conv_block(encoder1_pool, 128) # center
    decoder1 = decoder_block(center, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 64)
    drops = layers.Dropout(0.1)(decoder0)
    outputs = layers.Conv2D(output_band_count, (1, 1), activation='linear')(drops)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=opt, # optimizers.get(optimizer), 
        loss=losses.get(loss),
        metrics=[metrics.get(m) for m in show_metrics])

    return model
    


# Fit the model to the training data.
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights every 5 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=25*BATCH_SIZE)

model = get_model()
# model.save_weights(checkpoint_path.format(epoch=0))


batch_size = 4  # 196 is ok
train_datasets = FileSequence(TRAIN_DATA, batch_size=batch_size, width=64, repeats=1)
validation_datasets = FileSequence(VALIDATION_DATA, batch_size=batch_size, width=64, repeats=1)
with tf.device('/GPU:0'):
    model.fit(x=train_datasets, validation_data=validation_datasets, epochs=2)

model.save(MODEL_PATH, save_format='tf')


reader = TiffReader()
filename = '/data/Alarm/Samples/2021.05.01/147f298ebe5311ec81430242ac110003/probas_.tif'
with tf.device('/CPU:0'):
    print('Try file:', filename)
    dt, _, _ = reader.get_sample_image(filename)
    # crop input data to conform model
    conv_layer_count = 2  # layers in the model
    dx, dy = dt.shape[0], dt.shape[1]
    dx = (dx // 2**conv_layer_count) * 2**conv_layer_count
    dy = (dy // 2**conv_layer_count) * 2**conv_layer_count
    dt = dt[:dx, :dy, :]
    res = model.predict(np.array([dt]))
    res = np.squeeze(res)
    tiff.imwrite('/data/Alarm/Samples/2021.05.01/147f298ebe5311ec81430242ac110003/res.tiff', res, planarconfig='contig')
