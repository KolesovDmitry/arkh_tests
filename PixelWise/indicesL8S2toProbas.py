# -*- coding: utf-8 -*-

import os

from pprint import pprint

import tensorflow as tf
from tensorflow import keras


BUCKET = 'nextgis_gee_avral'

# Specify names locations
FOLDER = 'data'
TRAINING_BASE = 'trainSampleIdx'
EVAL_BASE = 'testSampleIdx'

TRAIN_FILE_PATH = os.path.join(FOLDER, TRAINING_BASE+'.tfrecord.gz')
TEST_FILE_PATH = os.path.join(FOLDER, EVAL_BASE+'.tfrecord.gz')

MODEL_DIR = 'models'
MODEL_NAME = 'model_v0_0_9'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


BANDS = ['fitted_EVI', 'fitted_NDVI', 'fitted_GNDVI', 'fitted_NRGI', 'fitted_RVI', 'fitted_NDWI', 'fitted_SAVI', 'EVI', 'NDVI', 'GNDVI', 'NRGI', 'RVI', 'NDWI', 'SAVI', 'dnum'] 
TARGET = 'target'
TARGET_BANDS = [TARGET]
FEATURE_NAMES = BANDS + TARGET_BANDS

# List of fixed-length features, all of which are float32.
COLUMNS = [tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES]
# Dictionary with names as keys, features as values.
FEATURES_DICT = dict(zip(FEATURE_NAMES, COLUMNS))

def parse_tfrecord(example_proto):
  """The parsing function.

  Read a serialized example into the structure defined by featuresDict.

  Args:
    example_proto: a serialized Example.

  Returns:
    A tuple of the predictors dictionary and the label, cast to an `int32`.
  """
  parsed_features = tf.io.parse_single_example(example_proto, FEATURES_DICT)
  targets = parsed_features.pop(TARGET)
  return parsed_features, targets

train_dataset = tf.data.TFRecordDataset(TRAIN_FILE_PATH, compression_type='GZIP')
# Map the function over the dataset.
train_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=5)



# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  
def to_tuple(inputs, targets):
    return (tf.expand_dims(tf.transpose(list(inputs.values())), 1), tf.expand_dims(targets, 1))

# TRAIN_SIZE = 725884
# EVAL_SIZE = 364127
BATCH_SIZE = 1024 * 32
BUFFER_SIZE = BATCH_SIZE

# Map the to_tuple function, shuffle and batch.
input_dataset = train_dataset.map(to_tuple).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)  # .repeat()
test_dataset = tf.data.TFRecordDataset(TEST_FILE_PATH, compression_type='GZIP').map(parse_tfrecord, num_parallel_calls=5).map(to_tuple).batch(BATCH_SIZE)  # .repeat()


# Define the layers in the model.
regularization_w = 0.0001
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((None, None, len(BANDS),)),
    tf.keras.layers.Conv2D(64, (1,1), activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.L2(regularization_w)),
    tf.keras.layers.GaussianNoise(0.02),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (1,1), activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.L2(regularization_w)),
    tf.keras.layers.GaussianNoise(0.02),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(regularization_w))
])

# Compile the model with the specified loss function.
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.metrics.binary_crossentropy, metrics=['mae'])
              

# Fit the model to the training data.
model.fit(x=input_dataset, validation_data=test_dataset, epochs=150)

model.save(MODEL_PATH, save_format='tf')
