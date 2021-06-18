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
MODEL_NAME = 'model_v0_0_3'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


BANDS = ['fitted_EVI', 'fitted_NDVI', 'fitted_GNDVI', 'fitted_NRGI', 'fitted_RVI', 'fitted_NDWI', 'fitted_SAVI', 'EVI', 'NDVI', 'GNDVI', 'NRGI', 'RVI', 'NDWI', 'SAVI'] 
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
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.
def to_tuple(inputs, targets):
  # return (tf.transpose(list(inputs.values())), targets.values())
  return (tf.transpose(list(inputs.values())), targets)

# Map the to_tuple function, shuffle and batch.
input_dataset = train_dataset.map(to_tuple).batch(1024*16)
test_dataset = tf.data.TFRecordDataset(TEST_FILE_PATH, compression_type='GZIP').map(parse_tfrecord, num_parallel_calls=5).map(to_tuple).batch(1024*16)


# Define the layers in the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with the specified loss function.
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')
              

# Fit the model to the training data.
model.fit(x=input_dataset, validation_data=test_dataset, epochs=500)

model.save(MODEL_PATH, save_format='tf')
