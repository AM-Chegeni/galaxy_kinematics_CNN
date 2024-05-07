import numpy as np
import os
import pandas as pd
import tqdm
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import random
import skimage._shared.utils #import channel_as_last_axis
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ActivityRegularization
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


X_train = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/X_train.npy')
X_test = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/X_test.npy')
y_train = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/y_train.npy')
y_test = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/y_test.npy')
prd1 = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/prd.npy')

from tensorflow.keras.models import load_model

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


best_model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])

filepath='/mn/stornext/u3/hassanif/amir/SAMI_project/model1/weights_best_model_200epoch.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-7)
    
history = best_model.fit(X_train, y_train,
          epochs=50,callbacks=[model_checkpoint_callback,reduce_lr],
          validation_data=(X_test, y_test))

np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/my_history.npy',history.history)

print("end of training")