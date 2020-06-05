from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import pydicom as dcm
import pandas as pd
import os

# Helper function to plot the dicom images
def plot_dicom_images(data, train_class):
  img_data = list(data.T.to_dict().values())
  f, ax = plt.subplots(3, 3, figsize = (16, 18))
  for i, row in enumerate(img_data):
      image = row['patientId'] + '.dcm'
      path = os.path.join('stage_2_train_images/', image)
      data = dcm.read_file(path)
      rows = train_class[train_class['patientId'] == row['patientId']]
      
      age = rows.PatientAge.unique().tolist()[0]
      sex = data.PatientSex
      part = data.BodyPartExamined
      vp = data.ViewPosition
      modality = data.Modality
      data_img = dcm.dcmread(path)
      ax[i//3, i%3].imshow(data_img.pixel_array, cmap = plt.cm.bone) 
      ax[i//3, i%3].axis('off')
      ax[i//3, i%3].set_title('ID: {}\nAge: {}, Sex: {}, Part: {}, VP: {}, Modality: {}\nTarget: {}, Class: {}\nWindow: {}:{}:{}:{}'\
                              .format(row['patientId'], age, sex, part, 
                                      vp, modality, row['Target'], 
                                      row['class'], row['x'], 
                                      row['y'], row['width'],
                                      row['height']))
      
      
      box_data = list(rows.T.to_dict().values())
      for j, row in enumerate(box_data):
        ax[i//3, i%3].add_patch(Rectangle(xy = (row['x'], row['y']),
                      width = row['width'], height = row['height'], 
                      color = 'blue', alpha = 0.15)) 
  plt.show()

# Helper function to get additional features from dicom images
def get_tags(data, path):
    images = os.listdir(path)
    for _, name in tqdm_notebook(enumerate(images)):
        img_path = os.path.join(path, name)
        img_data = dcm.read_file(img_path)
        idx = (data['patientId'] == img_data.PatientID)
        data.loc[idx,'PatientSex'] = img_data.PatientSex
        data.loc[idx,'PatientAge'] = pd.to_numeric(img_data.PatientAge)
        data.loc[idx,'BodyPartExamined'] = img_data.BodyPartExamined
        data.loc[idx,'ViewPosition'] = img_data.ViewPosition
        data.loc[idx,'Modality'] = img_data.Modality