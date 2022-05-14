import gdal
import sys
import os.path

import numpy as np

import tensorflow as tf
from keras import models


def read_model(model_path):
    return models.load_model(model_path)


def apply_model(filename, result_filename, model_path, input_band_count=8):
    import ipdb; ipdb.set_trace()
    model = read_model(model_path)

    in_ds = gdal.Open(filename)
    arr = np.array(in_ds.ReadAsArray())

    # roll axis to conform TF model input
    arr = np.rollaxis(arr, 0, 3)
    data = arr[:, :, :input_band_count]

    # crop input data to conform model
    conv_layer_count = 2  # layers in the model
    dx, dy = data.shape[0], data.shape[1]
    dx = (dx // 2**conv_layer_count) * 2**conv_layer_count
    dy = (dy // 2**conv_layer_count) * 2**conv_layer_count
    data = data[:dx, :dy, :]

    with tf.device('/CPU:0'):
        res = model.predict(np.array([data]))

    res = np.squeeze(res)

    # roll axis back to conform gdal dataset
    res = np.rollaxis(res, 2, 0)
    output_band_count = res.shape[0]

    projection   = in_ds.GetProjection()
    geotransform = in_ds.GetGeoTransform()

    driver = gdal.GetDriverByName('Gtiff')
    dataset = driver.Create(result_filename, dx, dx, output_band_count, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.WriteRaster(0, 0, dx, dy, res.tobytes())
    dataset.FlushCache()

    dataset = None
    in_ds = None
    


if __name__ == "__main__":
    model_path = '/tmp/Model'
    filename = '/data/Alarm/Samples/2021.09.01/0b511354cb9611ec8cb90242ac110008/probas_.tif'
    result_filename = '/data/Alarm/Samples/2021.09.01/0b511354cb9611ec8cb90242ac110008/result.tif'
    apply_model(filename, result_filename, model_path)



