# Information
A groundwork CNN trained using the Reptile meta learning framework from OpenAI. This project is focused on building a reliable few-shot recognition system for amputee patients sEMG signals, allowing for quick adaption to new bionic prosthetics. 

The project uses the NinaPro DB2 for training, and DB3 for testing. These datasets can be downloaded from the link in the Datasets folder.

## Phase I & II 

There are two phases to this project: the **unaugmented** and **augmented** phases. The augmented phases uses the same ACEnet model, however preprocessing was used to generate augmented sEMG samples to expand the size of the dataset.
This is especially useful for sEMG based classification tasks due to the small amount of sEMG data available; due to the amount of time and resources required to record a large number of said samples.

The Phase II test was included to see if the sample size makesa a difference to accuracy or not.

# Usage

Firstly use *pip install -r requirements.txt* to install **requirements.txt**.

Once you have the datasets organised and libraries downloaded, and the code has been updated to reflect the database destinations, you can run **ReptileCNN.py** to train the model using Reptile. The Meta-Learning framework is built in to this file.

Note that there is already a selection of saved models in the **Final_Models** folder.

## Testing

This project is tested in a Few-Shot environment to replicate that of a new amputee who may provide, say 5-10 sEMG recordings.
There is a **fewshottesting** and a **fewshotaugmented** .py file for each respective phase.

The **StandardCNN.py** and **AugmentedStandardCNN.py** are example CNN models that have been trained in the typical way, without using Reptile or any form of meta-learning. The Reptile results are compared to these models.

## Analysis

The **analysis.py** file is for analysing previously saved training outputs from the model. You can create a directory for your outputs and change the path to that directory, and the results will be plotted using *matplotlib*.

# Project Breakdown


