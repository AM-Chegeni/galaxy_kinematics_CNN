# import necessary packages

try:
    from keras_tuner import RandomSearch
    from keras_tuner.engine.hyperparameters import HyperParameters
except:
    !pip install keras-tuner
    from keras_tuner import RandomSearch
    from keras_tuner.engine.hyperparameters import HyperParameters

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


zip_path = '/mn/stornext/u3/hassanif/amir/SAMI_project/sami.zip'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/mn/stornext/u3/hassanif/amir/SAMI_project/new_sami_data')
    
    
def fits_data(directory):
    ids = []
    data1=[]

    # Get a list of FITS files in the directory
    fits_files = [file for file in os.listdir(directory) if file.endswith('.fits')]

    # Iterate over each FITS file
    for fits_file in fits_files:
        # Construct the full path to the FITS file
        fits_path = os.path.join(directory, fits_file)

        # Open the FITS file
        hdulist = fits.open(fits_path)

        # Access the data in the primary HDU
        data = hdulist[0].data

        # Close the FITS file
        hdulist.close()
        
            
        ids.append(int(fits_file.split('_')[0]))
        data1.append(data)

        print(f"File: {fits_file}")
    
    return (ids, data1)

data = fits_data('/mn/stornext/u3/hassanif/amir/SAMI_project/new_sami_data/sami/')
print(len(data[1]))

np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/new_sami_data/sami_data.npy',data[1])
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/new_sami_data/sami_label.npy',data[0])