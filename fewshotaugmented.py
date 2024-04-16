import os
# Force CPU usage (Enable if GPU not good enough)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# IMPORTS --------------------------------------------------------------------------------------------------------------------------------------

import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import NinaPro_Utility as npu
import numpy as np

# Get paths to files
path = os.path.abspath(__file__)
PROJ_DIRECTORY = os.path.dirname(path)
PROJ_DIRECTORY = PROJ_DIRECTORY.replace('\\', '/')
MODEL_DIRECTORY = f"{PROJ_DIRECTORY}/Final_Models/"
STANDARD_DIRECTORY = f"{PROJ_DIRECTORY}/Final_Models/"

# This is the testing & evaluation suite for the augmented and non-augmented built reptile models.
# The models are tested on unseen, new data, taken from the NinaPro DB3, which consists of sEMG signals from 11 amputees, providing
# a completely different task & scenario for the models to perform in. The overall goal of this research is to work towards
# providing bionic prosthesis for amputees, hence this dataset has been chosen.

# Both models are tested in a 'Few Shot' environment, to achieve our hypothesis. This is a 7-way 5-shot test, meaning that each
# model is shown 5 random training examples of the amputee data, and must then classify it as one of 7 gestures.

# The accuracies are likely to be very low due to being trained on raw, complex data, and tested on unseen data from an entirely new type
# of subject.

# Output for data. Full results will be saved to a .txt file in the package directory.

output_list = []
wincount = 0

# Retrieve the data and format it in to a dataframe for each subject
for subj in range(1, 12):
    indata = npu.get_data(f'{PROJ_DIRECTORY}/Datasets/NinaPro DB2/DB3_s{subj}', f'S{subj}_E1_A1.mat')

    # The gesture repetitions to train on
    train_reps = [1,3,4,6]

    # The gestue repetitions to test on
    test_reps = [2,5]

    # Normalise the training data using scikit standardscaler
    data = npu.normalise(indata, train_reps)
    testdata =  npu.normalise(indata, test_reps)

    # List for the 7 gesture ID's
    gestures = [i for i in range(1,8)]

    # Use Windowing to extract and build the dataset
    # Set window length and stride in ms
    win_len = 200
    win_stride = 20

    # Build the train and test sets from the data with the set parameters
    X_train, y_train, r_train = npu.windowing(data, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = npu.windowing(testdata, test_reps, gestures, win_len, win_stride)


    print(y_train.shape)
    print(X_train.shape)

    # Make a few shot example set

    # Set the number of examples per class
    num_examples_per_class = 5

    # Get the unique classes in y_train
    classes = np.unique(y_train)

    # Initialize the mini datasets
    x_train_mini = []
    y_train_mini = []

    # Loop over each class
    for c in classes:
        # Get the indices of the examples for this class
        indices = np.where(y_train == c)[0]
        
        # Randomly select num_examples_per_class indices
        selected_indices = np.random.choice(indices, size=num_examples_per_class, replace=False)
        
        # Add the selected examples to the mini datasets
        x_train_mini.append(X_train[selected_indices])
        y_train_mini.append(y_train[selected_indices])

    # Convert the mini datasets to numpy arrays
    x_train_mini = np.concatenate(x_train_mini)
    y_train_mini = np.concatenate(y_train_mini)


    # Convert to one hot
    y_train = npu.get_categorical(y_train)
    y_test = npu.get_categorical(y_test)

    y_train_mini = npu.get_categorical(y_train_mini)

    # AUGMENTED REPTILE MODEL --------------------------------------------------------------------------------------------------------------------------

    # Load the reptile model
    reptile_model = tf.keras.models.load_model(f'{MODEL_DIRECTORY}/augmented_checkpoint_reptile_model.h5')

    print(f"Testing Augmented Reptile Model on Subject {subj}...")

    # Show the unseen 5 examples of data to the reptile model
    reptile_model.fit(x_train_mini, y_train_mini)
    reptile_result = reptile_model.evaluate(X_test, y_test)
    reptile_result = reptile_result[1]


    # NORMAL REPTILE MODEL -------------------------------------------------------------------------------------------------------------------------

    # Load the standard model
    model = tf.keras.models.load_model(F'{STANDARD_DIRECTORY}/reptile_model_v1.h5')

    print(f"Testing Normal Reptile Model on Subject {subj}...")

    # Show the unseen 5 examples of data to the standard model
    model.fit(x_train_mini, y_train_mini)
    result = model.evaluate(X_test, y_test)
    result = result[1]


    # Compare and log results
    print('\n')
    print(f'Augmented subject {subj} accuracy = {reptile_result}%')
    print(f'Standard subject {subj} accuracy = {result}%')
    print('\n')

    # Count the amount of times Reptile outperformed the standard model
    if reptile_result > result:
        wincount += 1

    # Add the same data to a list for file writing
    output_list.append("\n")
    output_list.append(f"Subject {subj} Results")
    output_list.append(f'Augmented Reptile Model Accuracy: {reptile_result}%')
    output_list.append(f'Standard Reptile Model Accuracy: {result}%') 
    output_list.append("\n")


print(f'\nAugmented reptile model achieved higher accuracy than the standard reptile model on {wincount}/11 occasions\n')
output_list.append(f'\nAugmented reptile model achieved higher accuracy than the standard reptile model on {wincount}/11 occasions')

# Log the test to a txt file
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%d-%m-%Y_%H-%M")
with open(f'{MODEL_DIRECTORY}/augmented_amputee_testing/DB3_test_{timestamp}.txt', 'w') as f:
    # Write each element of the list to the file
    for item in output_list:
        f.write(f'{item}\n')

print('Test saved to file.')