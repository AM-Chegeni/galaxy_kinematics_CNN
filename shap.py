# import necessary packages
import numpy as np
import os
import pandas as pd
import tqdm
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import random
#import skimage._shared.utils #import channel_as_last_axis
import zipfile
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses
import tensorflow as tf
import os
import pandas as pd
import tqdm
import glob
from astropy.io import fits
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import tensorflow as tf
import timeit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import shap
    
# load numpy array and ids of SAMI survey
x_test=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/X_test.npy')
y_test=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/y_test.npy')
prd = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/prd.npy')
x_train=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/X_train.npy')
y_train=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/y_train.npy')

best_model = Sequential()
best_model.add(Conv2D(224, (3, 3), activation='relu', input_shape=(40, 40, 1)))
best_model.add(MaxPooling2D(pool_size=(1, 1)))
#best_model.add(Dropout(0.25))  # Add dropout layer with a rate of 0.25
best_model.add(Conv2D(160, (3, 3), activation='relu'))
best_model.add(MaxPooling2D(pool_size=(1, 1)))
#best_model.add(Dropout(0.25))  # Add dropout layer with a rate of 0.25
best_model.add(Conv2D(64, (3, 3), activation='relu'))
best_model.add(MaxPooling2D(pool_size=(1, 1)))
#best_model.add(Dropout(0.25))  # Add dropout layer with a rate of 0.25
best_model.add(Conv2D(224, (3, 3), activation='relu'))
best_model.add(MaxPooling2D(pool_size=(1, 1)))
#best_model.add(Dropout(0.25))  # Add dropout layer with a rate of 0.25
best_model.add(Flatten())
best_model.add(Dense(64, activation='relu'))
best_model.add(Dropout(0.5))  # Add dropout layer with a rate of 0.5
best_model.add(Dense(1, activation='sigmoid'))

#best_model.add(ActivityRegularization(l2=0.01))  # Add L2 regularization with a factor of 0.01

best_model.summary()


best_model.load_weights("/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/weights_best_model_200epoch.h5")
best_model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

explainer = shap.DeepExplainer(best_model,(x_train))

shap_values = explainer.shap_values(x_test)

shap_val = np.array(shap_values)
print(shap_val.shape)

shap_val = np.squeeze(shap_val)
print(shap_val.shape)

np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/results/shap_values.npy',shap_val)


print("finish):")


