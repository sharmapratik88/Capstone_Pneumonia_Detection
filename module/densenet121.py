# MODEL IMPORTS
import tensorflow as tf
assert tf.__version__ >= '2.0'

from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import DenseNet121
from keras.optimizers import Adam
import pydicom as dcm, cv2
import pandas as pd
import numpy as np
import keras

# Augmentation on the training data
train_datagen = ImageDataGenerator(rotation_range = 20, 
                                  width_shift_range = 0.2, 
                                  height_shift_range = 0.2, 
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# generators for memory efficient batch processing
def train_generator(df_train, IMAGE_SIZE, BATCH_SIZE):
        while True:
            for start in range(0, len(df_train), BATCH_SIZE):
                x_batch = []
                y_batch = []
                end = min(start + BATCH_SIZE, len(df_train))
                df_train_batch = df_train[start:end]
                for file, target in df_train_batch.values:
                    ds = dcm.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
                    img = np.stack((img,) * 3, -1)
                    img = train_datagen.random_transform(img)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

def valid_generator(df_valid, IMAGE_SIZE, BATCH_SIZE):
        while True:
            for start in range(0, len(df_valid), BATCH_SIZE):
                x_batch = []
                y_batch = []
                end = min(start + BATCH_SIZE, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for file, target in df_valid_batch.values:
                    ds = dcm.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

def pred_generator(df_valid, IMAGE_SIZE, BATCH_SIZE):
        while True:
            for start in range(0, len(df_valid), BATCH_SIZE):
                x_batch = []
                y_batch = []
                end = min(start + BATCH_SIZE, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for file, target in df_valid_batch.values:
                    ds = dcm.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch

def test_pred_generator(df_test, IMAGE_SIZE, BATCH_SIZE):
        while True:
            for start in range(0, len(df_test), BATCH_SIZE):
                x_batch = []
                y_batch = []
                end = min(start + BATCH_SIZE, len(df_test))
                df_test_batch = df_test[start:end]
                for file, target in df_test_batch.values:
                    ds = dcm.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch

def test_generator(df_test, IMAGE_SIZE, BATCH_SIZE):
        while True:
            for start in range(0, len(df_test), BATCH_SIZE):
                x_batch = []
                y_batch = []
                end = min(start + BATCH_SIZE, len(df_test))
                df_test_batch = df_test[start:end]
                for file, target in df_test_batch.values:
                    ds = dcm.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

# learning rate epochs              
def lrate_epoch(epoch):
   epochs_arr = [0, 30, 35, 40]
   learn_rates = [1e-5, 1e-6, 1e-7]
   lrate = learn_rates[0]
   if (epoch > epochs_arr[len(epochs_arr)-1]):
           lrate = learn_rates[len(epochs_arr)-2]
   for i in range(len(epochs_arr)-1):
       if (epoch > epochs_arr[i] and epoch <= epochs_arr[i+1]):
           lrate = learn_rates[i]
   return lrate

# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 20
    return lr*(np.cos(np.pi*x/epochs)+1.)/2

# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis = [1, 2, 3])
    union = tf.reduce_sum(y_true, axis = [1, 2, 3]) + tf.reduce_sum(y_pred, axis = [1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))