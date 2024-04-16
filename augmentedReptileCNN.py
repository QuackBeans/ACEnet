# IMPORTS ----------------------------------------------------------------------------------------------------------------------------------
import os
# Force CPU usage (Enable if GPU not good enough)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Optional terminal debugging (its spammy, so set to False unless issues aise)
import tensorflow as tf
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)

# Libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU,  BatchNormalization
import matplotlib.pyplot as plt
import NinaPro_Utility as npu
import random
import numpy as np
import gc
import re
from scipy.interpolate import interp1d

# Get paths to files
path = os.path.abspath(__file__)
PROJ_DIRECTORY = os.path.dirname(path)
PROJ_DIRECTORY = PROJ_DIRECTORY.replace('\\', '/')
MODEL_DIRECTORY = f"{PROJ_DIRECTORY}/Final_Models"

# Initialize an empty list to hold the tasks for each subject
tasks = []

# Function to split the subject's data in to tasks for Reptile
def create_tasks(x_train, y_train, x_test, y_test, k):
    # Create a list for the tasks
    subject_tasks = []

    
    # Get the unique gesture labels
    gesture_labels = np.unique(y_train)

    
    # Split the data into k tasks
    for _ in range(k):
        # Create empty lists to hold the training data for this task
        x_train_task = []
        y_train_task = []
        
        # For each gesture, create a task with 5 examples
        for label in gesture_labels:

            # Get the indices of the examples with this label
            label_indices = np.where(y_train == label)[0]

            # Randomly select 5 examples with this label
            selected_indices = np.random.choice(label_indices, size=5, replace=False)
            
            # Add the selected examples to the training data for this task
            x_train_task.extend(x_train[selected_indices])
            y_train_task.extend(y_train[selected_indices])
        

        # Convert the training data for this task to numpy arrays
        x_train_task = np.array(x_train_task)
        y_train_task = np.array(y_train_task)

        
        # Create a task for this subset of the data
        task = (x_train_task, y_train_task, x_test, y_test)

        # Add the task to the list of tasks
        subject_tasks.append(task)
    
    return subject_tasks


# Window Warping function
def window_warping(window_data, warping_factor):
    warped_window = np.copy(window_data)  # Create a copy of the original window

    window_size = window_data.shape[0]

    # Apply warping to the window
    warped_indices = np.arange(window_size) * warping_factor

    # Interpolate the warped indices
    warped_values = np.interp(warped_indices, range(window_size), window_data)
    warped_window = warped_values

    return warped_window


# Cycle through subjects 1 - 20
for subj in range(1, 21):

    # Load the data for each subject
    indata = npu.get_data(f'{PROJ_DIRECTORY}/Datasets/NinaPro DB2/DB2_s{subj}', f'S{subj}_E1_A1.mat')

    # The gesture repetitions to train on
    train_reps = [1,3,4,6]

    # The gestue repetitions to test on
    test_reps = [2,5]

    # Normalise the training data using scikit standardscaler
    data = npu.normalise(indata, train_reps)
    testdata = npu.normalise(indata, test_reps)

    # List for the 7 gesture ID's
    gestures = [i for i in range(1,8)]

    # Use Windowing to extract and build the dataset
    # Set window length and stride in ms
    win_len = 200
    win_stride = 20

    # Build the train and test sets from the data with the set parameters
    X_train, y_train, r_train = npu.windowing(data, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = npu.windowing(testdata, test_reps, gestures, win_len, win_stride)

    # List to store the warped windows and labels
    X_warped_data = []
    y_warped_data = []

    for win, lab in zip(X_train, y_train):

        # Apply window warping to augment each window then add it to the dataset, effectively producing 2x the amount of samples

        # Define the window size and warping factor
        window_size = 30
        warping_factor = 0.75

        # Copy the original data segment
        data_segment_copy = np.copy(win)

        # Randomly select a window
        start_index = np.random.randint(0, data_segment_copy.shape[0] - window_size)
        window = data_segment_copy[start_index:start_index+window_size, :]

        # Calculate the new window size after warping
        new_window_size = int(window_size * warping_factor)

        # Generate the new time axis for the warped window
        new_time_axis = np.linspace(0, 1, new_window_size)

        # Interpolate the data points in the window to introduce the additional points
        interpolated_window = np.zeros((new_window_size, window.shape[1]))
        for i in range(window.shape[1]):
            interp_func = interp1d(np.linspace(0, 1, window_size), window[:, i])
            interpolated_window[:, i] = interp_func(np.linspace(0, 1, new_window_size))

        # Replace the window in the copied data segment with the warped window
        data_segment_copy[start_index:start_index+new_window_size, :] = interpolated_window

        X_warped_data.append(data_segment_copy)
        y_warped_data.append(lab)


    X_warped_data = np.array(X_warped_data)
    y_warped_data = np.array(y_warped_data)

    X_train = np.concatenate([X_train, X_warped_data])
    y_train = np.concatenate([y_train, y_warped_data])

    # Shuffle the data to stop it from learning the patterns between the augmented and original data
    num_samples = X_train.shape[0]

    # Create an index array for shuffling
    indices = np.arange(num_samples)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Shuffle x_train and y_train using the shuffled indices
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Convert to one hot
    y_train = npu.get_categorical(y_train)
    y_test = npu.get_categorical(y_test)

    # Create k tasks for this subject
    subject_tasks = create_tasks(X_train, y_train, X_test, y_test, k=1000)

    # Add the k tasks for this subject to the tasks list
    tasks.extend(subject_tasks)    

    print(f"processed Subject {subj}")



# Create a validation set of completely new data

# Load the data for the Subject 40, who has been chosen randomly to build the validation set
data = npu.get_data(f'{PROJ_DIRECTORY}/Datasets/NinaPro DB2/DB2_s40', 'S40_E1_A1.mat')

# The gesture repetitions to train on
train_reps = [1,3,4,6]

# The gestue repetitions to test on
test_reps = [2,5]

# Normalise the training data using scikit standardscaler
data = npu.normalise(data, test_reps)

# List for the 7 gesture ID's
gestures = [i for i in range(1,8)]

# Use Windowing to extract and build the dataset
# Set window length and stride in ms
win_len = 200
win_stride = 20

# Build the test set from the data with the set parameters
X_val, y_val, r_val = npu.windowing(data, test_reps, gestures, win_len, win_stride)

# Convert to one hot
y_val = npu.get_categorical(y_val)

print(f"Built validation set from subject 40. X Shape: {X_val.shape}, Y Shape: {y_val.shape}")

# BUILD MODEL --------------------------------------------------------------------------------------------------------------------------


# Proposed ACEnet model architecture
def create_model():
    current_iter = 0 # initialise current training iteration as 0
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
    return model, current_iter


# Load Model from trained checkpoint (this allows training to be done in separate sessions)
def load_checkpoint():

    # Load the checkpoint model
    model = tf.keras.models.load_model(f'{MODEL_DIRECTORY}/augmented_versions/augmented_checkpoint_reptile_model.h5')
    
    # Load the last iteration
    with open(f"{MODEL_DIRECTORY}/augmented_versions/augmented_current_iteration.txt", "r") as f:
        # Read the current iteration from the file
        current_iter = int(f.read())

    return model, current_iter


# Lists to store inner loop task metrics
test_accuracy_list = []
test_loss_list = []

# Lists to store the metrics to plot after training is complete
task_train_accuracy_list = []
task_train_loss_list = []
task_test_accuracy_list = []
task_test_loss_list = []

# Define a dictionary that maps list names to lists (accuracies save to persistent disk to save memory)
lists = {
    "test_accuracy_list": test_accuracy_list,
    "test_loss_list": test_loss_list,
    "task_train_accuracy_list": task_train_accuracy_list,
    "task_train_loss_list": task_train_loss_list,
    "task_test_accuracy_list": task_test_accuracy_list,
    "task_test_loss_list": task_test_loss_list,
}

# TRAINING/REPTILE --------------------------------------------------------------------------------------------------------------------------

# Training loop function
def train_model_on_task(model, task, iteration):
    # Unpack the task data
    x_train, y_train, X_test, y_test = task # build data and label train sets from the random tuple chosen from tasks list
    
    # Compile the model
    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['accuracy'])
    
    # Train the model on the task data
    history = model.fit(x_train, y_train, batch_size=32, shuffle=True, verbose=1, validation_data = (X_test, y_test))
    

    # Extract the training metrics
    task_train_loss_list.append(history.history['loss'])
    task_train_accuracy_list.append(history.history['accuracy'])
    
    # Extract the validation metrics
    task_test_loss_list.append(history.history['val_loss'])
    task_test_accuracy_list.append(history.history['val_accuracy'])

    return model


# The Reptile meta-learning function. With each iteration it produces a fresh CNN model, applies the latest 'best' weights init to it,
# performs k steps of SGD on a random task from the task list and then uses its resulting learned parameters to
# u pdate the main model's initial parameters. 
# This is repeated for num_iterations until the parameters are at a point where they can generalise faster to new, unseen training/test data.

# Initialize empty lists to hold the validation accuracies for each patient

def reptile(model_fn, train_fn, tasks, num_iterations, k, epsilon):
    """
    model_fn: takes the function that creates a new instance of the model
    train_fn: takes the function that takes the model and a random task, and returns a trained model for that task
    tasks: the list of tasks
    num_iterations: the number of iterations to run the Reptile algorithm for
    k: the number of steps of SGD to perform on each task
    epsilon: the step size for the Reptile update
    """
    # Create an instance of the model with the current iteration of training its up to
    model, current_iter = model_fn()
    
    # Get the initial parameters of the model
    initial_params = model.get_weights()

    # Set meta batch size
    meta_batch_size = 8

    # Run full reptile training
    for iteration in range(current_iter, num_iterations):
        # Sample a meta batch of tasks from the tasks list
        task_indices = random.sample(range(len(tasks)), meta_batch_size)
        meta_batch = [tasks[i] for i in task_indices]

        # Initialize a list to store the final parameters for each task in the meta batch
        final_params_list = []

        # Process each task in the meta batch
        for task in meta_batch:
            # Create a new instance of the model
            task_model, current_iter = model_fn()

            # Set the initial parameters of the task model to the current best initial parameters
            task_model.set_weights(initial_params)

            # Train the task model on the task for k steps of SGD
            print(f"Task Training Iteration {iteration}")
            for _ in range(k):
                # Train on the training data from the task
                train_fn(task_model, task, iteration)
            
            # Get the new parameters of the task model after training on the task
            final_params = task_model.get_weights()
            
            # Add final parameters to list
            final_params_list.append(final_params)

        # Compute average final parameters across all tasks in meta batch
        avg_final_params = [sum(params) / len(params) for params in zip(*final_params_list)]
        
        # Update initial parameters using Reptile update sum with average final parameters
        initial_params = [initial + epsilon * (final - initial) for initial, final in zip(initial_params, avg_final_params)]
        
        # Set initial parameters of main model to updated initial parameters
        model.set_weights(initial_params)

        # Compile main model for evaluation
        opt_adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['accuracy'])

        # evaluate reptile model at each step
        print("\nReptile Model Validation:")
        loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
        print("\n")

        # string version of accuracy to use with filenames
        straccuracy = str(accuracy)

        # Extract the validation metrics
        test_loss_list.append(loss)
        test_accuracy_list.append(accuracy)

        # Save the metric lists to be opened again later and examined by iterating over the dictionary and write each list to a
        # separate text file
        for name, lst in lists.items():
            with open(f"{MODEL_DIRECTORY}/augmented_versions/augmented_{name}.txt", "a") as f:
                for item in lst:
                    f.write(f"{item}\n")


        # clear the lists at this iteration once they've been appended to the file to save memory
        test_accuracy_list.clear()
        test_loss_list.clear()
        task_train_accuracy_list.clear()
        task_train_loss_list.clear()
        task_test_accuracy_list.clear()
        task_test_loss_list.clear()


        # Save current iteration
        with open(f"{MODEL_DIRECTORY}/augmented_versions/augmented_current_iteration.txt", "w") as f:
            # Write the current iteration to the file
            f.write(f"{iteration}\n")


        # Try to save the checkpoint model
        try:
            # save the current model checkpoint to continue training at a later date
            model.save(f'{MODEL_DIRECTORY}/augmented_versions/augmented_checkpoint_reptile_model.h5')               
        except:
            print(f"SAVE ERROR: Save failed at this checkpoint, continuing without saving.")



        # Save the current most accurate validation model
        mod_directory = f'{MODEL_DIRECTORY}/augmented_versions/'
        indic = 'best_'


        # Find the latest model file
        exists = False
        for root, dirs, files in os.walk(mod_directory):
            for filename in files:
                if indic in filename:
                    exists = True
                    model_name = str(filename)
                    model_full_path = os.path.join(root, filename)

                    # Check if the latest model val_accuracy is better
                    oldacc = re.search(r'best_(\d+\.\d+)', filename)
                    if oldacc:
                        oldacc = float(oldacc.group(1))

                    if float(accuracy) >= oldacc:
                        print(f"new best model taken at iteration {iteration}: {round(accuracy, 4)}")
                        os.remove(model_full_path)
                        model.save(f'{MODEL_DIRECTORY}/augmented_versions/augmented_checkpoint_reptile_model_best_{straccuracy}.h5')
                    else:
                        pass

                else:
                    pass
              
        # If the best model file doesnt exist, create it
        if not exists:
            print("model checkpoint doesn't exist yet, creating file")
            model.save(f'{MODEL_DIRECTORY}/augmented_versions/augmented_checkpoint_reptile_model_best_{straccuracy}.h5')



        # Free up memory
        gc.collect()


    return model


# Run the Reptile algorithm on the model using the tasks defined and save it, or load a previous checkpoint in training to run from.
# Ask user to select to continue previous training or start fresh
valid_inputs = {'load', 'new', 'LOAD', 'NEW'}
loadyesno = ''

while loadyesno not in valid_inputs:
    loadyesno = input("\nLoad checkpoint model or start fresh? Enter: load/new\n")
    if loadyesno not in valid_inputs:
        print("Invalid input\n")

if loadyesno.lower() == 'load':
    print("Loading model...")
    model = reptile(load_checkpoint, train_model_on_task, tasks, num_iterations=10000, k=1, epsilon=0.1)

    # Final model save
    model.save(f'{MODEL_DIRECTORY}/augmented_versions/augmented_reptile_model.h5')
else:
    # Remove the current metric save files if they exist
    try:
        os.remove(f'{MODEL_DIRECTORY}/augmented_versions/augmented_test_accuracy_list.txt')
        os.remove(f'{MODEL_DIRECTORY}/augmented_versions/augmented_test_loss_list.txt')
        os.remove(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_train_accuracy_list.txt')
        os.remove(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_train_loss_list.txt')   
        os.remove(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_test_accuracy_list.txt')     
        os.remove(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_test_loss_list.txt')
    except:
        pass   
    # Create a new instance
    model = reptile(create_model, train_model_on_task, tasks, num_iterations=10000, k=1, epsilon=0.1)

    # Final model save
    model.save(f'{MODEL_DIRECTORY}/augmented_versions/augmented_reptile_model.h5')


# EVALUATION ----------------------------------------------------------------------------------------------------------------

def read_file_to_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

full_test_accuracy_list = read_file_to_list(f'{MODEL_DIRECTORY}/augmented_versions/augmented_test_accuracy_list.txt')
full_test_accuracy_list = [float(x.strip('[]')) for x in full_test_accuracy_list]

full_test_loss_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}/augmented_versions/augmented_test_loss_list.txt'))
full_test_loss_list = [float(x.strip('[]')) for x in full_test_loss_list]

full_task_train_accuracy_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_train_accuracy_list.txt'))
full_task_train_accuracy_list = [float(x.strip('[]')) for x in full_task_train_accuracy_list]

full_task_train_loss_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_train_loss_list.txt'))
full_task_train_loss_list = [float(x.strip('[]')) for x in full_task_train_loss_list]

full_task_test_accuracy_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_test_accuracy_list.txt'))
full_task_test_accuracy_list = [float(x.strip('[]')) for x in full_task_test_accuracy_list]

full_task_test_loss_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}/augmented_versions/augmented_task_test_loss_list.txt'))
full_task_test_loss_list = [float(x.strip('[]')) for x in full_task_test_loss_list]

# Plot task accuracy and task test
plt.figure(figsize=(10, 6))
plt.plot(full_task_train_accuracy_list, label='Task Train Accuracy')
plt.plot(full_task_test_accuracy_list, label='Task Validation Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Task Accuracy vs. Validation Accuracy')
plt.legend()
plt.show()

# Plot task train loss and task test loss
plt.figure(figsize=(10, 6))
plt.plot(full_task_train_loss_list, label='Task Train Loss')
plt.plot(full_task_test_loss_list, label='Task Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Train Loss vs. Validation Loss')
plt.legend()
plt.show()

# Create a figure with two subplots for the main model results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot Final Model Accuracy
ax1.plot(full_test_accuracy_list, label='Main Model Validation Accuracy', color='#ff7f0e')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Value')
ax1.set_title('Reptile Model Accuracy')
ax1.legend()

# Plot Final Model Loss
ax2.plot(full_test_loss_list, label='Main Model Validation Loss', color='#ff7f0e')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Value')
ax2.set_title('Reptile Model Loss')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()