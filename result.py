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



# load numpy array and ids of SAMI survey
x_test=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/X_test.npy')
y_test=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/y_test.npy')
prd = np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/prd.npy')




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



def confusion(pred, y_test, threshold):
    confusion_matrix_result = confusion_matrix(y_test, np.where(pred > threshold, 1, 0))
    FP = confusion_matrix_result[0][1]
    TN = confusion_matrix_result[0][0]
    TP = confusion_matrix_result[1][1]
    FN = confusion_matrix_result[1][0]
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    return tpr, fpr

def confusion(pred,y_test,threshold):
    confusion = confusion_matrix(y_test,np.where(pred > threshold, 1, 0))
    #print(confusion)
    FP = confusion[0][1]
    TN = confusion[0][0] 
    TP = confusion[1][1]
    FN = confusion[1][0]
    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    return tpr, fpr

import tqdm

def model_load(X_test, y_test, iteration):
    x = []
    y = []
    best_model.load_weights("/mn/stornext/u3/hassanif/amir/SAMI_project/model1/best_model_4thoct/weights_best_model_200epoch.h5")
    best_model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    mc_predictions1 = []
    for i in tqdm.tqdm(range(iteration)):
        y_p = best_model.predict(X_test)
        y_p = y_p.squeeze(1)
        #y_p = y_p.reshape(-1, 1)
        print(y_p.shape)
        mc_predictions1.append(y_p)
    mean = np.mean(mc_predictions1, axis=0)
    std = np.std(mc_predictions1, axis=0)

    for i in np.linspace(0.001, 0.999, 999):
        x1 = confusion(mean, y_test, i)[0]
        y1 = confusion(mean, y_test, i)[1]
        x.append(x1)
        y.append(y1)

    return mc_predictions1, mean, std, x, y


predictions_roc, mean_roc, std_roc, x_roc, y_roc = model_load(x_test,y_test,50)



np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/predictions_roc.npy',predictions_roc)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/mean_roc.npy',mean_roc)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/std_roc.npy',std_roc)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/x_roc.npy',x_roc)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/y_roc.npy',y_roc)




plt.figure(figsize=(8, 5))


#plt.bar(org_lrg_compact,alpha = 0.5,color = "b" ,linestyle='dashed',lw=1,label='Vanilla', edgecolor = 'black', align='center')
plt.hist(mean_roc,bins = 50,alpha = 0.5,color = "b",linestyle='dotted', lw=2,label='CNN ', edgecolor = 'black')
plt.hist(y_test,bins = 50,alpha = 0.5,color = "r",linestyle='dashed', lw=2,label='Real', edgecolor = 'black')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#plt.xscale('log')
#plt.xticks(np.round(y_dense,6)[998])
plt.xlabel('Prediction',size=25)
plt.ylabel('Number of Samples',size=25)
#plt.ylim(0,0.3)

plt.yscale('log')
plt.legend(loc='upper center',fontsize = 20)
plt.tight_layout()
plt.savefig('/mn/stornext/u3/hassanif/amir/SAMI_project/results/mean_dist.pdf', bbox_inches='tight')
#plt.show()




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"

plt.figure(figsize=(12, 6))
markersize = 7

roc_auc = roc_auc_score(y_test, mean_roc)

plt.plot(y_roc, x_roc,'b', marker='*', lw=1.5, markersize=markersize, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.xlabel('FP rate',size=27)
plt.ylabel('TP rate',size=27)
#plt.xlim(0,0.025)
plt.xticks(rotation = 90) 
plt.annotate('%.1e'%y_roc[998], xy=(np.round(y_roc,6)[998], np.round(x_roc,6)[998]), xytext=(np.round(y_roc,6)[998], np.round(x_roc,6)[998]),
             fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')


#plt.annotate(np.round(y_b3,6)[998], xy=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]), xytext=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]),
 #            fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.annotate(np.round(y_b3_dense121,6)[998], xy=(np.round(y_b3_dense121,6)[998], np.round(x_b3_dense121,6)[998]), xytext=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.annotate(np.round(y_dense121,6)[998], xy=(np.round(y_dense121,6)[998], np.round(x_dense121,6)[998]), xytext=(np.round(y_dense121,6)[998], np.round(x_dense121,6)[998]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

plt.xscale('log')

#plt.xlim(1e-4,0.1)
#plt.ylim(0.87,1)
#plt.title('ROC (Vanilla model)',fontsize = 25)
#plt.ylim(0,0.7)
plt.grid(True)
plt.legend(loc='lower right',fontsize = 13,fancybox=True,title_fontsize=16)
plt.savefig('/mn/stornext/u3/hassanif/amir/SAMI_project/results/roc.pdf', bbox_inches='tight')
plt.show()



history=np.load('/mn/stornext/u3/hassanif/amir/SAMI_project/model1/my_history.npy',allow_pickle='TRUE').item()




import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"

plt.figure(figsize=(12, 6))
markersize = 7


plt.plot(history['loss'],'b',label="Train", lw=1.5, markersize=markersize)
plt.plot(history['val_loss'],'r',label="Validation", lw=1.5, markersize=markersize)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.xlabel('Epoch',size=27)
plt.ylabel('Loss',size=27)
#plt.xlim(0,0.025)
#plt.xticks(rotation = 90) 
#plt.annotate('%.1e'%y_roc[98], xy=(np.round(y_roc,6)[98], np.round(x_roc,6)[98]), xytext=(np.round(y_roc,6)[98], np.round(x_roc,6)[98]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')


#plt.annotate(np.round(y_b3,6)[998], xy=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]), xytext=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]),
 #            fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.annotate(np.round(y_b3_dense121,6)[998], xy=(np.round(y_b3_dense121,6)[998], np.round(x_b3_dense121,6)[998]), xytext=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.annotate(np.round(y_dense121,6)[998], xy=(np.round(y_dense121,6)[998], np.round(x_dense121,6)[998]), xytext=(np.round(y_dense121,6)[998], np.round(x_dense121,6)[998]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.xscale('log')

#plt.xlim(1e-4,0.1)
#plt.ylim(0.87,1)
#plt.title('ROC (Vanilla model)',fontsize = 25)
#plt.ylim(0,0.7)
plt.grid(True)
plt.legend(loc='upper right',fontsize = 23,fancybox=True,title_fontsize=16)
plt.savefig('/mn/stornext/u3/hassanif/amir/SAMI_project/results/loss.pdf', bbox_inches='tight')
#plt.show()



import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"

plt.figure(figsize=(12, 6))
markersize = 7


plt.plot(history['accuracy'],'b',label="Train", lw=1.5, markersize=markersize)
plt.plot(history['val_accuracy'],'r',label="Validation", lw=1.5, markersize=markersize)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.xlabel('Epoch',size=27)
plt.ylabel('Accuracy',size=27)
#plt.xlim(0,0.025)
#plt.xticks(rotation = 90) 
#plt.annotate('%.1e'%y_roc[98], xy=(np.round(y_roc,6)[98], np.round(x_roc,6)[98]), xytext=(np.round(y_roc,6)[98], np.round(x_roc,6)[98]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')


#plt.annotate(np.round(y_b3,6)[998], xy=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]), xytext=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]),
 #            fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.annotate(np.round(y_b3_dense121,6)[998], xy=(np.round(y_b3_dense121,6)[998], np.round(x_b3_dense121,6)[998]), xytext=(np.round(y_b3,6)[998], np.round(x_b3,6)[998]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.annotate(np.round(y_dense121,6)[998], xy=(np.round(y_dense121,6)[998], np.round(x_dense121,6)[998]), xytext=(np.round(y_dense121,6)[998], np.round(x_dense121,6)[998]),
             #fontsize=15, ha='center', va='top', xycoords='data',textcoords='offset points')

#plt.xscale('log')

#plt.xlim(1e-4,0.1)
#plt.ylim(0.87,1)
#plt.title('ROC (Vanilla model)',fontsize = 25)
#plt.ylim(0,0.7)
plt.grid(True)
plt.legend(loc='lower right',fontsize = 23,fancybox=True,title_fontsize=16)
plt.savefig('/mn/stornext/u3/hassanif/amir/SAMI_project/results/accuracy.pdf', bbox_inches='tight')
#plt.show()


troubeling_fp=np.where((mean_roc>0.99) & (y_test.squeeze(1)==0))[0]
troubeling_fp.shape

troubeling_fn=np.where((mean_roc<0.99)&(y_test.squeeze(1)==1))[0]
troubeling_fn.shape

troubeling_tp=np.where((mean_roc>0.99)&(y_test.squeeze(1)==1))[0]
troubeling_tp.shape

troubeling_tn=np.where((mean_roc<0.99)&(y_test.squeeze(1)==0))[0]
troubeling_tn.shape


x_test[x_test == 0] = np.nan

y=mean_roc
prd=x_test
ind = troubeling_fp
for j in range(0,len(ind),5):
    
    fig,axes=plt.subplots(nrows=1,ncols=5,figsize=(20,20))
    
    axes[0].imshow(prd[ind[j]],cmap='jet')
    axes[0].set_title(f"CNN:{np.round(y[ind[j]],2)}\nReal:{np.round(y_test.squeeze(1)[ind[j]],2)}")
    axes[0].axes.get_xaxis().set_ticks([])
    axes[0].axes.get_yaxis().set_ticks([])
    cbar0=plt.colorbar(im0,ax=axes[0],fraction=0.046, pad=0.04)
    cbar0.ax.yaxis.set_ticks([])


    axes[1].imshow(prd[ind[j+1]],cmap='jet')
    axes[1].set_title(f"CNN:{np.round(y[ind[j+1]],2)}\nReal:{np.round(y_test.squeeze(1)[ind[j+1]],2)}")
    axes[1].axes.get_xaxis().set_ticks([])
    axes[1].axes.get_yaxis().set_ticks([])
    cbar1=plt.colorbar(im0,ax=axes[1],fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_ticks([])

    axes[2].imshow(prd[ind[j+2]],cmap='jet')
    axes[2].set_title(f"CNN:{np.round(y[ind[j+2]],2)}\nReal:{np.round(y_test.squeeze(1)[ind[j+2]],2)}")
    axes[2].axes.get_xaxis().set_ticks([])
    axes[2].axes.get_yaxis().set_ticks([])
    cbar2=plt.colorbar(im0,ax=axes[2],fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_ticks([])

    axes[3].imshow(prd[ind[j+3]],cmap='jet')
    axes[3].set_title(f"CNN:{np.round(y[ind[j+3]],1)}\nReal:{np.round(y_test.squeeze(1)[ind[j+3]],2)}")
    axes[3].axes.get_xaxis().set_ticks([])
    axes[3].axes.get_yaxis().set_ticks([])
    cbar3=plt.colorbar(im0,ax=axes[3],fraction=0.046, pad=0.04)
    #cbar3.ax.yaxis.set_ticks([])
    
    axes[4].imshow(prd[ind[j+4]],cmap='jet')
    axes[4].set_title(f"CNN:{np.round(y[ind[j+4]],2)}\nReal:{np.round(y_test.squeeze(1)[ind[j+4]],2)}")
    axes[4].axes.get_xaxis().set_ticks([])
    axes[4].axes.get_yaxis().set_ticks([])
    plt.colorbar(im0,ax=axes[4],fraction=0.046, pad=0.04)
    
    plt.savefig(f'/mn/stornext/u3/hassanif/amir/SAMI_project/results/lens_fp{j}.pdf', dpi=300, bbox_inches='tight')

    plt.show()
    
    
y=mean_roc
prd=x_test
ind = troubeling_fn
for j in range(0,len(ind),4):
    
    fig,axes=plt.subplots(nrows=1,ncols=4,figsize=(20,20))
    
    axes[0].imshow(prd[ind[j]],cmap='jet')
    axes[0].set_title(f"CNN:{round(y[ind[j]],12)}\nReal:{np.round(y_test.squeeze(1)[ind[j]],2)}")
    axes[0].axes.get_xaxis().set_ticks([])
    axes[0].axes.get_yaxis().set_ticks([])
    cbar0=plt.colorbar(im0,ax=axes[0],fraction=0.046, pad=0.04)
    cbar0.ax.yaxis.set_ticks([])


    axes[1].imshow(prd[ind[j+1]],cmap='jet')
    axes[1].set_title(f"CNN:{np.round(y[ind[j+1]],2)}\nReal:{np.round(y_test.squeeze(1)[ind[j+1]],2)}")
    axes[1].axes.get_xaxis().set_ticks([])
    axes[1].axes.get_yaxis().set_ticks([])
    cbar1=plt.colorbar(im0,ax=axes[1],fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_ticks([])

    axes[2].imshow(prd[ind[j+2]],cmap='jet')
    axes[2].set_title(f"CNN:{np.round(y[ind[j+2]],2)}\nReal:{np.round(y_test.squeeze(1)[ind[j+2]],2)}")
    axes[2].axes.get_xaxis().set_ticks([])
    axes[2].axes.get_yaxis().set_ticks([])
    cbar2=plt.colorbar(im0,ax=axes[2],fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_ticks([])

    axes[3].imshow(prd[ind[j+3]],cmap='jet')
    axes[3].set_title(f"CNN:{np.round(y[ind[j+3]],1)}\nReal:{np.round(y_test.squeeze(1)[ind[j+3]],2)}")
    axes[3].axes.get_xaxis().set_ticks([])
    axes[3].axes.get_yaxis().set_ticks([])
    cbar3=plt.colorbar(im0,ax=axes[3],fraction=0.046, pad=0.04)
    #cbar3.ax.yaxis.set_ticks([])
    
    plt.savefig(f'/mn/stornext/u3/hassanif/amir/SAMI_project/results/lens_fn{j}.pdf', dpi=300, bbox_inches='tight')

    plt.show()



