# IMPORTS ----------------------------------------------------------------------------------------------------------------------------------

import os
import tensorflow as tf
# Optional terminal debugging (its spammy, so only activate when issues arise)
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)

import numpy as np
import matplotlib.pyplot as plt
import NinaPro_Utility as npu
import sklearn.metrics
from scipy.io import loadmat
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.models import Sequential, Model, load_model, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
import json

# Get paths to files
path = os.path.abspath(__file__)
PROJ_DIRECTORY = os.path.dirname(path)
MODEL_DIRECTORY = f"{PROJ_DIRECTORY}\\Final_Models"


# Lists for the combined data of the subjects
x_train_full = []
y_train_full = []
x_test_full = []
y_test_full = []

# PREPROCESSING ---------------------------------------------------------------------------------------------------------------------------

# Cycle through subjects 1 - 20
for i in range(1, 21):

    # Load the data for each subject
    indata = npu.get_data(f'{PROJ_DIRECTORY}/Datasets/NinaPro DB2/DB2_s{i}', f'S{i}_E1_A1.mat')

    # The gesture repetitions to train on
    train_reps = [1,3,4,6]

    # The gestue repetitions to test on
    test_reps = [2,5]

    # Normalise the training and test data
    data = npu.normalise(indata, train_reps)
    testdata = npu.normalise(indata, test_reps)

    # List for the 7 gesture ID's
    gestures = [i for i in range(1,8)]

    # Windowing the time series data
    # Set window length and stride in ms, as per the original Ninapro DB2 paper recommendation
    win_len = 200
    win_stride = 20

    # Build the train and test sets from the data with the window parameters
    X_train, y_train, r_train = npu.windowing(data, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = npu.windowing(testdata, test_reps, gestures, win_len, win_stride)

    # Convert to one hot
    y_train = npu.get_categorical(y_train)
    y_test = npu.get_categorical(y_test)

    # Append to the full dataset
    x_train_full.append(X_train)
    y_train_full.append(y_train)
    x_test_full.append(X_test)
    y_test_full.append(y_test)

    print(f"processed Subject {i}")


# Convert combined lists in to tensor shape
X_train = np.concatenate(x_train_full, axis=0)
y_train = np.concatenate(y_train_full, axis=0)
X_test = np.concatenate(x_test_full, axis=0)
y_test = np.concatenate(y_test_full, axis=0)

# Check shapes match and fit the network
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# VALIDATION SET ----------------------------------------------------------------------------------------------------------------------------
# Create a validation set of unseen data. This is for harsh benchmarking in preparation for the 'new task' data test.

# Load the data for the Subject 40, who has been chosen randomly to build the validation set from.
data = npu.get_data(f'{PROJ_DIRECTORY}/Datasets/NinaPro DB2/DB2_s40', 'S40_E1_A1.mat')

# The gesture repetitions to train on
train_reps = [1,3,4,6]

# The gestue repetitions to test on
test_reps = [2,5]

# Normalise the data
data = npu.normalise(data, test_reps)

# List for the 7 gesture ID's
gestures = [i for i in range(1,8)]

# indowing
# Set window length and stride in ms
win_len = 200
win_stride = 20  

# Build the test set from the data with the set parameters
X_val, y_val, r_val = npu.windowing(data, test_reps, gestures, win_len, win_stride)

# Convert to one hot
y_val = npu.get_categorical(y_val)

print(f"Built validation set from subject 40. X Shape: {X_val.shape}, Y Shape: {y_val.shape}")

# MODEL & TRAINING ----------------------------------------------------------------------------------------------------------------------------

# Proposed ACEnet model architecture
def get_model(X_train):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(200, 12, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

# Create the model
model = get_model(X_train) 


# Training loop that evaluates the model and includes early stopping for when the val_loss functions stops improving.
def train_model(model, X_train_wind, y_train_wind, X_test_wind, y_test_wind, save_to, epoch = 50):
        opt_adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['accuracy'])

        # checkpoints
        mc = ModelCheckpoint(save_to + 'standard_cnn_model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
        
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        history = model.fit(x=X_train_wind, y=y_train_wind, epochs=epoch, batch_size=32, shuffle=True,
                    verbose=1,
                    validation_data = (X_test_wind, y_test_wind), callbacks=[mc, es])
        
        # Save the model
        model.save(save_to + 'standard_cnn_model.h5')

        # Free up memory
        gc.collect()
        
        return history,model


# Train the model & record the metrics
history, model = train_model(model, X_train, y_train, X_val, y_val, save_to= f'{MODEL_DIRECTORY}/', epoch = 300)

# ANALYSIS ---------------------------------------------------------------------------------------------------------------------------------------

# Save the history to a file to be reopened and analysed

# Convert the history object to a dictionary
history_dict = history.history

# Save the history to a text file
with open('standard_model_history.txt', 'w') as f:
    json.dump(history_dict, f)

# Load the history from the text file
with open('history.txt', 'r') as f:
    history_dict = json.load(f)

# Plot training & validation accuracy values
plt.plot(history_dict.history['accuracy'])
plt.plot(history_dict.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test on Subj 40'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_dict.history['loss'])
plt.plot(history_dict.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test on Subj 40'], loc='upper left')
plt.show()


# Evaluate the model on the testing data from each subject. This test isn't as reliable as the unseen data, as it is formed from the subjects
# that were trained on.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test loss (on test data from trained subjects):", loss)
print("Test accuracy (on test data from trained subjects):", accuracy)

# Generate predictions on the testing data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded testing labels to integers
y_test_classes = np.argmax(y_test, axis=1)

# Calculate and print the confusion matrix
cm = sklearn.metrics.confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion matrix:\n", cm)

# Print report
cr = sklearn.metrics.classification_report(y_test_classes, y_pred_classes)
print("Classification report:\n", cr)