## RSNA Pneumonia Detection Detection - Capstone Project
Capstone Project completed as a part of Great Learning's PGP - Artificial Intelligence and Machine Learning.

## Problem Statement
In this capstone project, the goal is to build a pneumonia detection system, to locate the position of inflammation in an image. Tissues with sparse material, such as lungs which are full of air, do not absorb the X-rays and appear black in the image. Dense tissues such as bones absorb X-rays and appear white in the image. While we are theoretically detecting “lung opacities”, there are lung opacities that are not pneumonia related. In the data, some of these are labeled “Not Normal No Lung Opacity”. This extra third class indicates that while pneumonia was determined not to be present, there was nonetheless some type of abnormality on the image and oftentimes this finding may mimic the appearance of true pneumonia.

## Approach
### Step 1: Exploratory Data Analysis
* Understanding of the data with a brief on train/test labels and respective class info
* Look at the first five rows of both the csvs (train and test)
* Identify how are classes and target distributed
* Check the number of patients with 1, 2, ... bounding boxes
* Read and extract metadata from dicom files
* Perform analysis on some of the features from dicom files
* Check some random images from the training dataset
* Draw insights from the data at various stages of EDA

**Outcome**
* [EDA: Jupyter Notebook Link](https://nbviewer.jupyter.org/github/sharmapratik88/Capstone_Pneumonia_Detection/blob/master/CP_RSNA_Pneumonia_Detection_EDA.ipynb)
* [EDA: Module Link](https://github.com/sharmapratik88/Capstone_Pneumonia_Detection/blob/master/module_eda.py)
* [EDA: Output (pickle files)](https://github.com/sharmapratik88/Capstone_Pneumonia_Detection/tree/master/output)
