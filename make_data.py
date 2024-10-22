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

array_dict = {id_: data[i] for i, id_ in enumerate(label)}

def remove_doub_val_dic(data_dict):
    # Assuming `array_dict` is your dictionary with ids as keys
    # Step 1: Count occurrences of each id
    id_counts = Counter(data_dict.keys())

    # Step 2: Identify duplicates (ids that occur more than once)
    duplicate_ids = {id_ for id_, count in id_counts.items() if count > 1}

    # Step 3: Remove duplicates from the dictionary
    array_dict_non_doub = {id_: value for id_, value in data_dict.items() if id_ not in duplicate_ids}
    
    # Optional: if you want to print the number of duplicates found
    num_duplicates = len(duplicate_ids)
    print(f"Number of duplicated ids removed: {num_duplicates}")
    
    for key, value in array_dict_non_doub.items():
        # Use numpy to replace NaN with 0 in the 2D arrays
        array_dict_non_doub[key] = np.nan_to_num(value, nan=0)
    
    return array_dict_non_doub

new_dict = remove_doub_val_dic(array_dict)
len(new_dict)

def check_spin(spin):
    # Step 1: Identify duplicated CATID elements
    duplicates = spin[spin.duplicated(subset='CATID', keep=False)]

    # Step 3: For each group of duplicated CATID, handle according to your conditions
    for catid, group in duplicates.groupby('CATID'):
        # Check if all values in FR_EE11 and SR_EE11 are NaN
        if group[['FR_EE11', 'SR_EE11']].isna().all(axis=1).all():
            # Keep only one of the duplicated rows
            indices_to_keep = group.index[:1]  # Keep the first row
            indices_to_drop = group.index[1:]  # Drop the remaining rows
            # Drop the other rows from the DataFrame
            spin = spin.drop(indices_to_drop)
        
        # Check if FR_EE11 is 1.0 and SR_EE11 is NaN
        elif ((group['FR_EE11'] == 1.0) & (group['SR_EE11'].isna())).any():
            # Keep the row where FR_EE11 is 1.0 and SR_EE11 is NaN
            index_to_keep = group[(group['FR_EE11'] == 1.0) & (group['SR_EE11'].isna())].index
            indices_to_drop = group.index.difference(index_to_keep)
            # Drop the other rows from the DataFrame
            spin = spin.drop(indices_to_drop)

        # Check if SR_EE11 is 1.0 and FR_EE11 is NaN
        elif ((group['SR_EE11'] == 1.0) & (group['FR_EE11'].isna())).any():
            # Keep the row where SR_EE11 is 1.0 and FR_EE11 is NaN
            index_to_keep = group[(group['SR_EE11'] == 1.0) & (group['FR_EE11'].isna())].index
            indices_to_drop = group.index.difference(index_to_keep)
            # Drop the other rows from the DataFrame
            spin = spin.drop(indices_to_drop)

    # The resulting DataFrame 'spin' will have the label column and duplicates handled.
    return spin
    
spin1 = check_spin(spin)

# Assuming your DataFrame is named 'spin'
duplicates_spin = spin1[spin1['CATID'].duplicated(keep=False)]

# Create a new DataFrame with only the duplicate rows
df_duplicates = duplicates_spin.copy()

# Print the new DataFram
df_duplicates


def create_new_array_based_on_spin(spin, my_dict):
    new_array = []  # Initialize an empty list to store new values
    
    # Iterate through the keys of the dictionary (which are CATID values)
    for catid in my_dict.keys():
        # Check if the CATID exists in the spin['CATID'] column
        if catid in spin['CATID'].values:
            # Find the corresponding row(s) in spin DataFrame
            row = spin[spin['CATID'] == catid]
            
            # Take the first row if multiple exist
            if len(row) > 0:
                row = row.iloc[0]  # Take the first row if multiple are present
                
                # Extract scalar values
                fr_ee11 = row['FR_EE11']
                sr_ee11 = row['SR_EE11']
                
                # Check the conditions and append to the new_array
                if pd.isna(fr_ee11) and pd.isna(sr_ee11):
                    new_array.append(2)
                
                elif fr_ee11 == 1.0 and pd.isna(sr_ee11):
                    new_array.append(1)
                
                elif pd.isna(fr_ee11) and sr_ee11 == 1.0:
                    new_array.append(0)

    # Return the new array with appended values
    return new_array


def cut_dict_values(data_dict, cut):
    ll = 25 - cut
    rl = 25 + cut
    new_dict = {}
    
    for key, array in data_dict.items():
        if array.shape[0] > rl and array.shape[1] > rl:  # Ensure array is large enough
            new_dict[key] = array[ll:rl, ll:rl]
        else:
            print(f"Array for key '{key}' is too small to cut.")
    
    return new_dict


cut_dict = cut_dict_values(new_dict,20)
list(cut_dict.items())[0][1].shape



def filter_and_remove_items(data_dict, array, target_value):
    """
    Filters a dictionary to include only the items where the corresponding 
    array value equals `target_value`, and removes those items from the original dictionary.
    
    :param data_dict: Dictionary with values to be filtered and removed.
    :param array: The array of corresponding values.
    :param target_value: The value in the array that determines which dictionary 
                         items to keep and remove.
    
    :return: A new dictionary with filtered items, and the modified original dictionary.
    """
    new_dict = {}
    keys_to_remove = []

    # Iterate through the dictionary and array
    for i, (key, value) in enumerate(data_dict.items()):
        # If the array value is equal to target_value, add the item to new_dict
        if array[i] == target_value:
            new_dict[key] = value
            keys_to_remove.append(key)

    # Remove the filtered items from the original dictionary
    for key in keys_to_remove:
        del data_dict[key]

    return new_dict, data_dict


target_value = 2  # We want to filter by this value in the array

filtered_dict, updated_data_dict = filter_and_remove_items(cut_dict, new_array, target_value)


import pickle

# Save the dictionary using pickle
with open('unknown_samples.pkl', 'wb') as pickle_file:
    pickle.dump(filtered_dict, pickle_file)

print("Dictionary saved using pickle.")


def remove_value(array, value):
  """Removes all occurrences of a given value from an array.

  Args:
    array: The input array.
    value: The value to remove.

  Returns:
    A new array without the specified value.
  """

  return [x for x in array if x != value]


modified_array = remove_value(new_array, 2)
len(modified_array)


# Set up the data generator for augmentation
datagen = ImageDataGenerator(rotation_range=360, 
                             width_shift_range=4, 
                             height_shift_range=4, 
                             zoom_range=[1.0, 1.05], 
                             horizontal_flip=True, 
                             fill_mode='nearest')

def augment_and_expand(data_dict, array, special_value, num_augments=1, special_augments=7):
    """
    Augments dictionary values and adds new entries under the same key.
    The elements in the dictionary whose corresponding array value matches `special_value`
    will be augmented `special_augments` times. Other elements will be augmented `num_augments` times.
    
    :param data_dict: Dictionary with 2152 values to be augmented (each of shape (40, 40)).
    :param array: The array of size 2152 with corresponding values.
    :param special_value: The value in the array that triggers special augmentation.
    :param num_augments: Number of augmentations to perform for each dictionary entry (default: 1).
    :param special_augments: Number of augmentations to perform for the special entries.
    
    :return: A new expanded dictionary and array with augmentations applied.
    """
    new_data_dict = {}
    new_array = []

    # Initialize a counter to ensure unique keys for the augmented entries
    key_counter = 0

    # Iterate over each key and value in the dictionary
    for i, (key, value) in enumerate(data_dict.items()):
        # Ensure value shape is (40, 40)
        if value.shape == (40, 40):
            # Reshape to (1, 40, 40, 1) because the generator works with batches
            value = value.reshape((1, 40, 40, 1))

            # Determine number of augmentations based on the array value
            augments = special_augments if array[i] == special_value else num_augments

            # Add the original value first
            new_data_dict[f'{key}_{key_counter}'] = value[0].reshape((40, 40))
            new_array.append(array[i])
            key_counter += 1

            # Generate the specified number of augmentations
            for j in range(augments):
                aug_iter = datagen.flow(value, batch_size=1)
                aug_value = next(aug_iter)[0].reshape((40, 40))

                # Add the augmented value under the same key but with a unique suffix
                new_data_dict[f'{key}_{key_counter}'] = aug_value
                new_array.append(array[i])
                key_counter += 1

        else:
            print(f"Skipping key '{key}' due to incorrect shape.")

    return new_data_dict, np.array(new_array)

# Example usage
# Assume 'data_dict' is your dictionary and 'fixed_array' is the associated array
# Example: data_dict = {'item1': np.random.rand(40,40), 'item2': np.random.rand(40,40), ... up to 2152 keys}
# Example: fixed_array = np.random.rand(2152)  # Array with same length as data_dict

# Special value in the array for which the corresponding dictionary values should be augmented more
special_value = 0  # Example: augment elements where array has value 0

new_data_dict, sug_array = augment_and_expand(updated_data_dict, modified_array, special_value, num_augments=8, special_augments=68)

import random

def shuffle_dictionary_and_array(data_dict, array):
    """
    Shuffles the dictionary items while preserving the relationship with the array values.
    
    :param data_dict: Dictionary to be shuffled.
    :param array: Array corresponding to the dictionary keys.
    
    :return: A shuffled dictionary and the corresponding shuffled array.
    """
    # Convert the dictionary into a list of items (key-value pairs)
    items = list(data_dict.items())
    
    # Combine the dictionary items and array into a list of tuples
    combined = list(zip(items, array))
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Unzip the combined list back into shuffled dictionary items and shuffled array
    shuffled_items, shuffled_array = zip(*combined)
    
    # Rebuild the shuffled dictionary
    shuffled_dict = dict(shuffled_items)
    
    return shuffled_dict, np.array(shuffled_array)

# Example usage
# Assuming `data_dict` and `fixed_array` are already defined
shuffled_dict, shuffled_array = shuffle_dictionary_and_array(new_data_dict, sug_array)


X_train, X_test, y_train, y_test, train_label, test_label = total_data[:15532], total_data[15532:], shuffled_array[:15532], shuffled_array[15532:],\
total_label[:15532], total_label[15532:]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(train_label.shape)
print(test_label.shape)


np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_14sep/x_train.npy',X_train)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_14sep/y_train.npy',y_train)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_14sep/x_test.npy',X_test)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_14sep/y_test.npy',y_test)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_14sep/train_label.npy',train_label)
np.save('/mn/stornext/u3/hassanif/amir/SAMI_project/update_14sep/test_label.npy',test_label)






