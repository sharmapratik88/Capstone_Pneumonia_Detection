# Reference: https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components
import numpy as np, pandas as pd
import os, random, pydicom, keras
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import cv2

from skimage import io, measure
from skimage.transform import resize

from matplotlib import pyplot as plt
import matplotlib.patches as patches

PATH = '/content/drive/My Drive/Pneumonia_Detection/'
DATA_DIR = os.path.join(PATH + 'data/')

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

def get_pneumonia_evidence(filename):
    pneumonia_evidence = {}
    df = pd.read_csv(DATA_DIR + filename)

    for _, row in df.iterrows():
        fn = row[0]
        loc = row[1:5]
        pneumonia = row[5]
        if pneumonia == 1:
            loc = [int(float(i)) for i in loc]
            if fn in pneumonia_evidence:
                pneumonia_evidence[fn].append(loc)
            else:
                pneumonia_evidence[fn] = [loc]
    return pneumonia_evidence

class generator(keras.utils.Sequence):

    def __init__(self, folder, fns, pneumonia_evidence = None, 
                 batch_size = 32, image_size = 256, 
                 shuffle = True, augment = False, predict = False):
        self.folder = folder
        self.fns = fns
        self.pneumonia_evidence = pneumonia_evidence
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, fn):
        img = pydicom.dcmread(os.path.join(self.folder, fn)).pixel_array
        msk = np.zeros(img.shape)
        fn = fn.split('.')[0]
        if fn in self.pneumonia_evidence:
            for loc in self.pneumonia_evidence[fn]:
                x, y, w, h = loc
                msk[y:y+h, x:x+w] = 1
        img = resize(img, (self.image_size, self.image_size), mode = 'reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode = 'reflect') > 0.5
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk

    def __loadpredict__(self, fn):
        img = pydicom.dcmread(os.path.join(self.folder, fn)).pixel_array
        img = resize(img, (self.image_size, self.image_size), mode = 'reflect')
        img = np.expand_dims(img, -1)
        return img

    def __getitem__(self, index):
        fns = self.fns[index*self.batch_size:(index+1)*self.batch_size]

        if self.predict:
            imgs = [self.__loadpredict__(fn) for fn in fns]
            imgs = np.array(imgs)
            return imgs, fns

        else:
            items = [self.__load__(fn) for fn in fns]
            imgs, msks = zip(*items)
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.fns)

    def __len__(self):
        if self.predict:
            return int(np.ceil(len(self.fns)/self.batch_size))
        else:
            return int(len(self.fns)/self.batch_size)

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
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

# create network and compiler
model = create_network(input_size = 256, channels = 32, n_blocks = 2, depth = 4)
model.compile(optimizer = 'adam',
              loss = iou_bce_loss,
              metrics = ['accuracy', mean_iou])

# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr*(np.cos(np.pi*x/epochs)+1.)/2

# Function to plot masks that are generated using `generator` function above
def plot_masks(df, path, image_fns, pneumonia_evidence):
  sample_patient_id = random.choice(list(df.loc[(df['Target'] == 1), 'patientId']))
  sample_fn = sample_patient_id + '.dcm'
  sample_details = df.loc[df['patientId'] == sample_patient_id]

  g = generator(path, image_fns, pneumonia_evidence)
  img, msk = g.__load__(sample_fn)
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 6))

  _ = ax1.imshow(img[:,:,0], cmap = plt.cm.bone); _ = ax1.axis('off')
  _ = ax2.imshow(msk[:,:,0]); _ = ax2.axis('off')
  _ = ax3.imshow(cv2.bitwise_and(img, img, mask = msk.astype(np.uint8)), 
                cmap = plt.cm.bone); _ = ax3.axis('off')

  _ = ax1.set_title('{}\nAge: {}, Gender: {}, VP: {}\nSample Image'.format(sample_patient_id,
                list(sample_details['PatientAge'].unique())[0], 
                list(sample_details['PatientSex'].unique())[0],
                list(sample_details['ViewPosition'].unique())[0]))
  _ = ax2.set_title('{}\nAge: {}, Gender: {}, VP: {}\nMask for Sample Image'.format(sample_patient_id,
                list(sample_details['PatientAge'].unique())[0], 
                list(sample_details['PatientSex'].unique())[0],
                list(sample_details['ViewPosition'].unique())[0]))
  _ = ax3.set_title('{}\nAge: {}, Gender: {}, VP: {}\nMask overlay over Image'.format(sample_patient_id,
                list(sample_details['PatientAge'].unique())[0], 
                list(sample_details['PatientSex'].unique())[0],
                list(sample_details['ViewPosition'].unique())[0]))
  plt.subplots_adjust(top = 0.4)
  plt.tight_layout()