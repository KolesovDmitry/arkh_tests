# -*- coding: utf-8 -*-

import os

from pprint import pprint

import numpy as np

from tifffile import tifffile as tiff

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import models
from keras import losses
from keras import metrics
# from keras.utils import Sequence
from tensorflow.keras.utils import Sequence

# from keras.callbacks import TensorBoard
# from keras.callbacks import ModelCheckpoint

BUCKET = 'nextgis_gee_avral'

# Specify names locations
FOLDER = '/data/Alarm/Samples'
TRAINING_BASE = 'trainSampleIdx'
EVAL_BASE = 'testSampleIdx'

TRAIN_FILE_PATH = os.path.join(FOLDER, TRAINING_BASE+'.tif')
TEST_FILE_PATH = os.path.join(FOLDER, EVAL_BASE+'.tif')

MODEL_DIR = '/data/Alarm/Models'
MODEL_NAME = 'model_v0_0_01'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


INPUT_BANDS = ['day_num', 'evi', 'ndvi', 'gndvi', 'nrgi', 'rvi', 'ndwi', 'savi'] 
TARGET_BANDS = ['EVI', 'NDVI', 'GNDVI', 'NRGI', 'RVI', 'NDWI', 'SAVI']
FEATURE_NAMES = INPUT_BANDS + TARGET_BANDS


class FileSequence(Sequence):
    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
               for filename in batch_x]), np.array(batch_y)


    def shrade_ds(self, dataset, count=512, width=1024):
        dataset = get_sample_image('/data/Alarm/22.03.01/04c4b456b58d11ec912a0242ac110003/probas_.tif')
        x, y = dataset
        rows, cols = x.shape[0], x.shape[1]
        i = np.random.randint(low=0, high=rows-width, size=count)
        j = np.random.randint(low=0, high=cols-width, size=count)

        xdata = np.array(
            [x[i[k]: i[k]+width, j[k]: j[k]+width, :] for k in range(count)]
        )
        ydata = np.array(
            [y[i[k]: i[k]+width, j[k]: j[k]+width, :] for k in range(count)]
        )

        return xdata, ydata

    def get_sample_image(self, filename):
        input_band_count = 8
        x = tiff.imread(filename)
        x = np.nan_to_num(x)   
        y = x[:, :, input_band_count:]

        return (x[:, :, :input_band_count], y)


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
    outputs = layers.Conv2D(output_band_count, (1, 1), activation='linear')(decoder0)

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

# import ipdb; ipdb.set_trace()
# Define the layers in the model.
model = get_model()
# model.save_weights(checkpoint_path.format(epoch=0))



for i in range(100):
    print('Iteration:', i)
    x, y = shrade_ds(input_dataset, count=1024, width=64)
    with tf.device('/GPU:0'):
        model.fit(x=x, y=y, validation_split=0.85, epochs=30)

model.save(MODEL_PATH, save_format='tf')

with tf.device('/CPU:0'):
    dt = input_dataset[0]
    dt = dt[:4096, :4096, :]
    res = model.predict(np.array([dt]))
    res = np.squeeze(res)
    tiff.imwrite('/data/Alarm/22.03.01/04c4b456b58d11ec912a0242ac110003/res.tif', res, planarconfig='contig')
