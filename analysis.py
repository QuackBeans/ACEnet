import matplotlib.pyplot as plt
import os

# This file is for analysing the data from previous runs. Change the directory/filenames to plot them and compare.
# Currently set to augmented reptile run v1

# Get paths to files
path = os.path.abspath(__file__)
PROJ_DIRECTORY = os.path.dirname(path)
PROJ_DIRECTORY = PROJ_DIRECTORY.replace('\\', '/')
MODEL_DIRECTORY = f"{PROJ_DIRECTORY}/Final_Models/augmented-reptile_run_v1_data/"
def read_file_to_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

full_test_accuracy_list = read_file_to_list(f'{MODEL_DIRECTORY}augmented_test_accuracy_list.txt')
full_test_accuracy_list = [float(x.strip('[]')) for x in full_test_accuracy_list]

full_test_loss_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}augmented_test_loss_list.txt'))
full_test_loss_list = [float(x.strip('[]')) for x in full_test_loss_list]

full_task_train_accuracy_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}augmented_task_train_accuracy_list.txt'))
full_task_train_accuracy_list = [float(x.strip('[]')) for x in full_task_train_accuracy_list]

full_task_train_loss_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}augmented_task_train_loss_list.txt'))
full_task_train_loss_list = [float(x.strip('[]')) for x in full_task_train_loss_list]

full_task_test_accuracy_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}augmented_task_test_accuracy_list.txt'))
full_task_test_accuracy_list = [float(x.strip('[]')) for x in full_task_test_accuracy_list]

full_task_test_loss_list = read_file_to_list(os.path.join(f'{MODEL_DIRECTORY}augmented_task_test_loss_list.txt'))
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