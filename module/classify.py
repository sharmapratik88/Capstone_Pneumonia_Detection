# MODEL IMPORTS
import tensorflow as tf
assert tf.__version__ >= '2.0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
from imgaug import augmenters as iaa

import pydicom as dcm, cv2
from PIL import Image
import pandas as pd
import numpy as np
import keras
random_state = 2020

# ROC AUC as a Metric
# Reference: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def roc_auc(y_true, y_pred):
    return tf.compat.v1.py_function(roc_auc_score, (y_true, y_pred), tf.double)

# Average Precision as a Metric
# Reference: https://stackoverflow.com/questions/57918572/keras-metric-equivalent-to-scikit-learns-average-precision-score-metric
import tensorflow.keras.backend as K
def average_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Augmenter
augmenter = ImageDataGenerator(preprocessing_function = preprocess_input, 
                               rotation_range = 20, width_shift_range = 0.2,
                               height_shift_range = 0.2, zoom_range = 0.2,
                               horizontal_flip = True)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_df, img_dir, batch_size = 32, dim = (224, 224), 
                 transform = None, n_channels = 1, shuffle = True, debug = False):
        
        self.dataset_df = dataset_df
        self.paths = self.dataset_df['path']
        self.img_dir = img_dir 
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.debug = debug
        self.transform = transform
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataset_df) / self.batch_size))

    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]    

        list_IDs_batch = [self.paths[k] for k in indexes]

        imgs, labels = self.__data_generation(list_IDs_batch)

        if self.debug: return list_IDs_batch, imgs, labels 
        else: return imgs, labels
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset_df))
        if self.shuffle == True: np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_batch):
        imgs = np.empty((self.batch_size, *self.dim, self.n_channels))
        labels = np.empty((self.batch_size), dtype = int)
        for i, ID in enumerate(list_IDs_batch): 
            imgs[i,] = self.load_img(self.img_dir, ID, self.dim)
            labels[i] = self.get_label(ID)
        return imgs, labels
    
    def load_img(self, img_dir, path, dim):
        img = Image.open(path)
        img = np.asarray(img.convert('RGB'))
        img = img / 255.
        img = cv2.resize(img, dim)
        if not self.transform == None:
            params = self.transform.get_random_transform(img.shape)
            img = self.transform.apply_transform(img, params)
#         imagenet_mean = np.array([0.485, 0.456, 0.406])
#         imagenet_std = np.array([0.229, 0.224, 0.225])
#         img = (img - imagenet_mean) / imagenet_std
        return img
    
    def get_label(self, path):
        data_df = self.dataset_df.loc[self.dataset_df['path'] == path].values
        return int(data_df[0][1])

# Predictor
def predictor(model, validation_generator):
    y_pred = []
    y_roc = []
    y_true = []
    for i, batch in enumerate(validation_generator):
        imgs = batch[0]
        labels = batch[1]
        for img, label in zip(imgs, labels):
            img = np.expand_dims(img, axis = 0)
            pred = model.predict(img)
            y_roc.append(pred)
            if pred > 0.5: pred = 1
            else: pred = 0
            y_pred.append(pred)
            y_true.append(label)
    return y_true, y_pred, y_roc