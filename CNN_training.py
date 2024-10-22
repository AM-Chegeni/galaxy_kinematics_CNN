# import necessary packages
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import numpy as np
import os
import pandas as pd
import tqdm
import json
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
from tensorflow.keras.models import load_model
import random
import json
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers
from collections import Counter

X_train = np.load('/mn/stornext/u3/chegenia/SAMI_project/augmented_data/x_train.npy')
y_train = np.load('/mn/stornext/u3/chegenia/SAMI_project/augmented_data/y_train.npy')
X_test = np.load('/mn/stornext/u3/chegenia/SAMI_project/augmented_data/x_test.npy')
y_test = np.load('/mn/stornext/u3/chegenia/SAMI_project/augmented_data/y_test.npy')
train_label = np.load('/mn/stornext/u3/chegenia/SAMI_project/augmented_data/train_label.npy')
test_label = np.load('/mn/stornext/u3/chegenia/SAMI_project/augmented_data/test_label.npy')


X_train = np.expand_dims(X_train,axis=-1)
y_train = np.expand_dims(y_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
y_test = np.expand_dims(y_test,axis=-1)

print(X_train.shape)

"""
def build_model(hp):
    model = keras.Sequential()

    # Tune the number of convolutional layers
    for i in range(hp.Int('conv_layers', 1, 7)):  
        if i == 0:
            # Add the input shape only to the first layer
            model.add(keras.layers.Conv2D(
                filters=hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
                activation='relu',
                input_shape=(40, 40, 1),  # Only for the first layer
                kernel_regularizer=regularizers.l2(0.001)  # Add L2 regularization
            ))
        else:
            model.add(keras.layers.Conv2D(
                filters=hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)  # Add L2 regularization
            ))
        
        # Add Batch Normalization before MaxPooling
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Use (2, 2) for downsampling

    model.add(keras.layers.Flatten())
    
    # Tune the number of dense layers
    for i in range(hp.Int('dense_layers', 1, 2)):  
        model.add(keras.layers.Dense(
            units=hp.Int(f'dense_units_{i}', min_value=32, max_value=256, step=32),
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)  # Add L2 regularization
        ))
        # Use tunable dropout rate
        model.add(keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    # Output layer for binary classification
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Single output unit for binary classification

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')
        ),
        loss='binary_crossentropy',  # Loss function for binary classification
        metrics=['accuracy']
    )

    return model


# Tuner configuration
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='/mn/stornext/u3/hassanif/amir/SAMI_project/update_8oct',
    project_name='cnn_tuning'
)


# Search for the best model
tuner.search(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


# Assuming you've already done the tuning with the code above

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the entire model (architecture + weights)
best_model.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_7oct/best_cnn_model.h5')  # Saved in HDF5 format

# Save only the model weights
best_model.save_weights('/mn/stornext/u3/hassanif/amir/SAMI_project/update_7oct/best_cnn_weights.weights.h5')

# Save the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
hyperparams_dict = best_hyperparameters.values

# Save hyperparameters as JSON
with open('/mn/stornext/u3/hassanif/amir/SAMI_project/update_7oct/best_hyperparameters.json', 'w') as json_file:
    json.dump(hyperparams_dict, json_file)

print("Model, weights, and hyperparameters saved successfully......")
"""

def build_model():
    model = models.Sequential()
    
    # First Conv Layer
    model.add(layers.Conv2D(filters=64, kernel_size=3,activation="relu",input_shape=(40, 40, 1), kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())

    # Second Conv Layer
    model.add(layers.Conv2D(filters=128, kernel_size=3,activation="relu",kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())

# Second Conv Layer
    model.add(layers.Conv2D(filters=256, kernel_size=3,activation="relu",kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())

    # Flatten the output
    model.add(layers.Flatten())

    # First Dense Layer
    model.add(layers.Dense(units=96, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))  # Dropout with 40%

    # Second Dense Layer
    model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))  # Dropout with 40%

    # Output Layer for Binary Classification
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


best_model = build_model()
best_model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

best_model.compile(optimizer=Adam(learning_rate = 0.001
), loss='binary_crossentropy', metrics=['accuracy'])
# Define the ModelCheckpoint callback to save the best weights based on validation accuracy
# Define the ModelCheckpoint callback to save the entire model based on validation accuracy
checkpoint_callback = ModelCheckpoint(
    filepath='/mn/stornext/u3/chegenia/SAMI_project/cnn_model/best_weights.weights.h5',  # Path to save the full model
    monitor='val_accuracy',            # Monitor validation accuracy
    save_best_only=True,               # Save only when the validation accuracy improves
    save_weights_only=True,           # Save the entire model (architecture + weights)
    mode='max',                        # Save based on max validation accuracy
    verbose=1                          # Print status messages when saving
)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
# Train the model for 150 epochs, including the checkpoint callback
history = best_model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback,early_stopping]
)
best_model.save('/mn/stornext/u3/chegenia/SAMI_project/cnn_model/best_cnn_model.keras')  # Saved in HDF5 format

best_model.save('/mn/stornext/u3/chegenia/SAMI_project/cnn_model/best_cnn_model.h5')  # Saved in HDF5 format

# Convert history.history (a dictionary) to a DataFrame
history_df = pd.DataFrame(history.history)

# Save the DataFrame to a CSV file
history_df.to_csv('/mn/stornext/u3/chegenia/SAMI_project/cnn_model/training_history.csv', index=False)

print("Training history saved as CSV.")
