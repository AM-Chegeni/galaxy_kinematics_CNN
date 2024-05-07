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


######################################################

# load numpy array and ids of SAMI survey
data=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/new_sami_data/sami_data.npy')
label=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/new_sami_data/sami_label.npy')
spin = pd.read_csv("/mn/stornext/u3/hassanif/amir/SAMI_project/old_sami_data/SAMI/all_sami_data.csv")
print(data.shape, label.shape)
##################################################################

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

def make_data(data,label,spin):
    
    # find number of duplicated lables 
    dup = pd.DataFrame(label)[pd.DataFrame(label).duplicated([0],keep=False)]
    print(f"Number of duplicacated labels is:{dup.shape}")
    # delete these duplicated lables from lables
    d_fil = filter_rows_by_values(pd.DataFrame(label), 0, dup[0])

    #reset label indexing
    d_fil1 = d_fil.reset_index()

    #check for suring about repetitive lables
    q = d_fil[d_fil.duplicated([0],keep=False)] 
    print(f"number of duplicated lables in new label data : {q}")

    #sort label indexing
    d_fil.sort_index()

    # nan values == 0 in data files
    data[np.isnan(data)] = 0

    #apply not-duplicated lables on CATID column in SAMI csv data
    filter_spin = spin[spin['CATID'].apply(lambda x: x in np.array(d_fil))]
    filter_spin1 = filter_spin.insert(0, 'id', range(0, len(filter_spin)))

    # find duplicated lables in CATID column 
    df = filter_spin[filter_spin.duplicated(['CATID'],keep=False)]
    df1 = df.fillna(0)

    #delete duplicated lables in CATID column (these ids identified in code debugging)
    df2 = filter_rows_by_values(filter_spin, "id", [169,172,173,176,181,184,186,189,241,243,245,
    253,1438,1441,1502,1513,1628,1630,1632,1638,1711,1717,1722,1725])
    print(f"number of duplicated lables is : {len(df2)}")
    df3 = df2.fillna(0)

    print(f"data shape is : {data[0].shape}")

    return d_fil1,df3,data

def selection(label,df3,data):
    label1=[]
    data_new =[]
    prd = []
    for i in range(1966):
        print(i)
        lab = label[0][i]
        print(lab)
        h = df3.loc[df3['CATID']==lab]
        
        print(h.shape)
        if int(h['FR_EE11'])==1 and int(h['SR_EE11'])==0:
            print('fast')
            label1.append(1)
            data_new.append(data[i])
        elif int(h['FR_EE11'])==0 and int(h['SR_EE11'])==1:
            print('slow')
            label1.append(0)
            data_new.append(data[i])
        else:
            prd.append(data[i])

    data_new = np.array(data_new)
    label1 = np.array(label1)
    prd = np.array(prd)
    return data_new, label1, prd

def cut_frame(data,cut):
    j = 0
    ll=25-cut
    rl=25+cut
    new40 = []
    while j<data.shape[0]:     #source_train.shape[0]:

        new = data[j,:,:][ll:rl, ll:rl]
        new40.append(new)
        j +=1
    new40 = np.array(new40)
    print(new40.shape)
    return new40

def augmentation(data_new1, label1):
    new40new = []
    new_label = []
    
    datagen = ImageDataGenerator(rotation_range=360, fill_mode='nearest')

    #datagen = ImageDataGenerator(rotation_range=360, fill_mode='nearest')

    for i in range(data_new1.shape[0]):
        current = None  # Assign a default value to 'current'
        
        if label1[i] == 0:
            tmp = data_new1[i, :, :]
            aug_iter = datagen.flow(tmp.reshape(1, 40, 40, 1), batch_size=1)

            for k in range(7):
                current = next(aug_iter)[0].astype('float')
                new40new.append(current.reshape(40, 40))  # Reshape to 2D array
                new_label.append(label1[i])

        else:
            tmp = data_new1[i, :, :]
            new40new.append(tmp.reshape(40, 40))  # Reshape to 2D array
            new_label.append(label1[i])

    new40new = np.array(new40new)
    new_label = np.array(new_label)

    return new40new, new_label


###################################################################################

d_fil1,df3,data = make_data(data,label,spin)
data_new, label1, prd = selection(d_fil1,df3,data);
data_new1 = cut_frame(data_new,20)
prd1 = cut_frame(prd,20)
X_train_aug, y_train_aug = augmentation(data_new1,label1)
print(prd1.shape)
print(data_new1.shape)
print(label1.shape)

####################################################################################

print(X_train_aug.shape,y_train_aug.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train_aug,y_train_aug, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = np.expand_dims(X_train,axis=-1)
y_train = np.expand_dims(y_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
y_test = np.expand_dims(y_test,axis=-1)

X_train.shape

from numpy import save
# save to npy file
save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/X_train.npy', X_train)
save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/X_test.npy', X_test)
save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/y_train.npy', y_train)
save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/y_test.npy', y_test)
save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/prd.npy', prd1)


#########################################################