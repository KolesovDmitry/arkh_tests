# -*- coding: utf-8 -*-

'''This module is used for NN training only. It is nor required for inference of trained models'''


import os
import random
import sys

from pprint import pprint

import numpy as np

# from pathlib import Path 

import gdal

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import models
from keras import losses
from keras import metrics
from tensorflow.keras.utils import Sequence


# Specify names locations
FOLDER = '/data/Alarm/Samples/'
LANDSAT_FILENAME = 'landsat.tif'
SENTINEL_FILENAME = 'sentinel.tif'

VALIDATION_RATIO = 0.33 

INPUT_MODEL_PATH = '/data/Alarm/Samples/Init_Model'
OUTPUT_MODEL_PATH = '/data/Alarm/Samples/Final_Model'


class TiffReader:
    def get_sample_image(self, filename, return_mask=False):
        ds = gdal.Open(filename)
        arr = np.array(ds.ReadAsArray())
        arr = np.nan_to_num(arr)

        # roll axis to conform TF model input
        arr = np.rollaxis(arr, 0, 3)

        if return_mask:
            mask = np.logical_not(np.isnan(arr).any(axis=2))
            mask = mask.astype(int)
            arr = np.nan_to_num(arr)
            return arr, mask
        else:
            arr = np.nan_to_num(arr)
            return arr


class FileSequence(Sequence):
    def __init__(self, filenames, batch_size, repeats=32, width=1024):
        self.filenames = filenames
        random.shuffle(self.filenames)
        self.batch_size = batch_size
        self.width = width
        self.tiff_reader = TiffReader()

        self.iter_count = 0
        self.max_iter_count = repeats

    def __len__(self):
        return len(self.filenames) * self.max_iter_count

    def __getitem__(self, idx):
        return self.shrade_ds(idx)

    def shrade_ds(self, idx):
        idx = idx % len(self.filenames)
        x_file, y_file = self.filenames[idx]
        x = self.get_sample_image(x_file, return_mask=False)
        y, mask = self.get_sample_image(y_file, return_mask=True)
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

    def get_sample_image(self, filename, return_mask=False):
        # filename = str(filename)
        return self.tiff_reader.get_sample_image(filename, return_mask=return_mask)



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
    
############################################################
# Train process
############################################################

# Get list of PAIRS landsat/sentinel images
# Input bands (Landsat): ['day_num', 'evi', 'ndvi', 'gndvi', 'nrgi', 'rvi', 'ndwi', 'savi'] 
# target_bands (Sentinel): ['EVI', 'NDVI', 'GNDVI', 'NRGI', 'RVI', 'NDWI', 'SAVI']
DATA_FILES = []
for root, dirs, files in os.walk(FOLDER):
    if (LANDSAT_FILENAME in files) and (SENTINEL_FILENAME in files):
        DATA_FILES.append(
            (os.path.join(root, LANDSAT_FILENAME), os.path.join(root, SENTINEL_FILENAME))
        )

# SHuffle before train/test splitting
random.shuffle(DATA_FILES)

validation_count = int(len(DATA_FILES) * VALIDATION_RATIO)

TRAIN_DATA = sorted(DATA_FILES[validation_count:])
VALIDATION_DATA = sorted(DATA_FILES[:validation_count])


# Fit the model to the training data.
if len(sys.argv) == 1:
    model = get_model()
elif sys.argv[1] == 'retrain':
    model = models.load_model(INPUT_MODEL_PATH)
else:
    raise RuntimeError('Use "retrain" argument or none of arguments')

batch_size = 32   # 196 is ok for my GPU and file size
train_datasets = FileSequence(TRAIN_DATA, batch_size=batch_size, width=64, repeats=3)
validation_datasets = FileSequence(VALIDATION_DATA, batch_size=batch_size, width=64, repeats=3)
with tf.device('/GPU:0'):
    model.fit(x=train_datasets, validation_data=validation_datasets, epochs=10)

model.save(OUTPUT_MODEL_PATH, save_format='tf')


# Use apply_model.script

