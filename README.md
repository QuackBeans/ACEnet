# Information
A groundwork CNN trained using the Reptile meta learning framework from OpenAI. This project is focused on building a reliable few-shot recognition system for amputee patients sEMG signals, allowing for quick adaption to new bionic prosthetics. 

The project uses the NinaPro DB2 for training, and DB3 for testing. These datasets can be downloaded from the link in the Datasets folder.

## NinaPro DB2

This is the training data used in the study. It contains all of the intact subjects used for all of the models training, and also subject 40, the subject who’s data is used for the in-training validation.

## NinaPro DB3

This is the testing data used in the study for the few-shot environment. It contains the data of amputated subjects, to serve as non-ideal, harsh testing data.

## Phase I & II 

There are two phases to this project: the **unaugmented** and **augmented** phases. The augmented phases uses the same ACEnet model, however preprocessing was used to generate augmented sEMG samples to expand the size of the dataset.
This is especially useful for sEMG based classification tasks due to the small amount of sEMG data available; due to the amount of time and resources required to record a large number of said samples.

The Phase II test was included to see if the sample size makesa a difference to accuracy or not.

# Usage

Firstly use *pip install -r requirements.txt* to install **requirements.txt**.

Once you have the datasets organised and libraries downloaded, and the code has been updated to reflect the database destinations, you can run **ReptileCNN.py** to train the model using Reptile. The Meta-Learning framework is built in to this file.

Note that there is already a selection of saved models in the **Final_Models** folder, along with their respective training and validation data.

## Testing

This project is tested in a Few-Shot environment to replicate that of a new amputee who may provide, say 5-10 sEMG recordings.
There is a **fewshottesting** and a **fewshotaugmented** .py file for each respective phase.

The **StandardCNN.py** and **AugmentedStandardCNN.py** are example CNN models that have been trained in the typical way, without using Reptile or any form of meta-learning. The Reptile results are compared to these models.

## Analysis

The **analysis.py** file is for analysing previously saved training outputs from the model. You can create a directory for your outputs and change the path to that directory, and the results will be plotted using *matplotlib*.

# Project Breakdown

## requirements.txt

The code relies on a number of external libraries. 
This file provides all of the dependencies for the project.

To install them using pip, navigate to the project directory and type pip install -r requirements.txt in the command line/terminal.

If you are using Python3, pip should be installed automatically with it.

## StandardCNN.py

This is the python code for the standard CNN model’s preprocessing, training, evaluation and analysis.

## ReptileCNN.py

This is the python code for the ACEnet model’s preprocessing, training, evaluation and analysis.

## AugmentedStandardCNN.py

This is the python code for the **augmented** version of the standard CNN model’s preprocessing, training, evaluation and analysis.


## augmentedReptileCNN.py

This is the python code for the **augmented** version of ACEnet’s preprocessing, training, evaluation and analysis

## analysis.py

An extra python file for loading previously saved training/testing metrics and analysing them with matplotlib. The directories and filenames can be changed in the code to plot them and compare your results.

## NinaPro_Utility.py

A selection of functions useful for processing the NinaPro Datasets. Special thanks to @parasgulati8 for saving me time with these. The full set of processing functions can be found on his GitHub https://github.com/parasgulati8/NinaPro-Helper-Library.

## fewshottesting.py

This is the few-shot testing environment for the unaugmented models. It pitches ACEnet against the standard CNN in a 7-way 5-shot test.

## fewshotaugmented.py

This is the few-shot testing environment for the augmented models. It pitches augmented ACEnet against unaugmented ACEnet in a 7-way 5-shot test.

## README.md

This README.

## Datasets/README.md

The folder to contain the NinaPro datasets. You will need to adjust paths in the code according to how you organise your data once it's processed and cleaned.


## Final_Models

This folder contains multiple subfolders for each part of the experiment.

    The unaugmented training (Phase I)
    The augmented training (Phase II)

    The unaugmented amputee few-shot testing (Phase I)
    The augmented amputee few-shot testing (Phase II)


All of the final models have been organised in to data folders, however the original copies are also left in the parent folder for program access.

### standard_cnn_run_v1_data (folder)

This folder is a copy of the standard CNN model post-training. It can be loaded using tensorflow.


### reptile_run_v1_data (folder)

This folder contains all of the data from the Reptile CNN’s run. Due to the way Reptile’s training works with tasks, the metrics were recorded into text files to clear up memory throughout training, and can be loaded in from these files using the analysis.py file. It contains the metrics for all of the visualisations shown in the report, namely the task performance metrics and the overall ACEnet model performance metrics.


It also contains the final model at convergence, and a checkpoint of the best accuracy.

The current_iteration.txt file is for reloading and continuing training.


Full item list:


    README.txt
    checkpoint_reptile_model.h5
    reptile_model_v1.h5
    current_iteration.txt
    task_test_accuracy_list.txt
    task_test_loss_list.txt
    task_train_accuracy_list.txt
    task_train_loss_lost.txt
    test_accuracy_list.txt
    test_loss_list.txt


### augmented-standard_run_v1_data (folder)

This folder contains the augmented standard cnn model.


### augmented-reptile_run_v1_data (folder)

This folder contains all of the data from the augmented Reptile CNN’s run. Due to the once again the metrics were recorded into text files to clear up memory throughout training, and can be loaded in from these files using the analysis.py file. It contains the metrics for all of the visualisations shown in the report, namely the task performance metrics and the overall augmented ACEnet model performance metrics.


It also contains the final model at convergence, and a checkpoint of the best accuracy.


The augmented_current_iteration.txt file is for reloading and continuing training.


Full item list:


    README.txt
    augmented_checkpoint_reptile_model.h5
    augmented_reptile_model_v1.h5
    augmented_current_iteration.txt
    augmented_task_test_accuracy_list.txt
    augmented_task_test_loss_list.txt
    augmented_task_train_accuracy_list.txt
    augmented_task_train_loss_lost.txt
    augmented_test_accuracy_list.txt
    augmented_test_loss_list.txt


### amputee_testing (folder)

This is the folder that stores the output results from the few-shot amputee testing.

### augmented_amputee_testing (folder)

This is the folder that stores the output results from the augmented few-shot amputee testing. There is currently data of two previous runs in there which were used for the report.


### augmented_checkpoint_reptile_model.h5

### augmented_standard_cnn_model.h5

### checkpoint_reptile_model.h5

### reptile_model_v1.h5

### standard_cnn_model_v1.h5

# Final Notes

## CPU/GPU

At the top of the model code, there is an option to use either GPU or CPU, depending on the abilities of the computer. If one mode does not work, please try the other. The Reptile model is quite intensive and can take a long time to train.

## Future Work

As previously mentioned, this is a groundwork experiment, and foundational research. The aim here was to fill a gap in current sEMG recognition knowledge, by attempting a new framework implementation (Reptile Meta-learning). There is more work to be done, and more to be explored, as this is only the first version of this model. Any contributions, ideas and tips are highly encouraged and welcome to aid in increasing the feasability of the model.