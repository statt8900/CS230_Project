# CS230_Project
Project Goal:
Almost every chemical we interact with requires a catalyst to create. Today, the activity of catalysts is determined computationally using Density Functional Theory (DFT). Unfortunately, this method is extremely expensive and can be a research bottleneck.

The convolution operation has proven to be an excellent feature extractor in the world of computer vision. This one operation is crucial to a wide variety of image-processing tasks, such as image classification and segmentation. We plan to create an analogous operation that can be applied to surface science. This operation will enable machine learning algorithms to expedite catalyst design. The fundamental mapping DFT provides, which we wish to learn, is one from a chemical structure to its energy.
Guide to files:

CNN_input.py    - Creation of PyTorch Dataset from sqlite3 database

convolution.py  - Creation of PyTorch.nn Modules for chemical convolution   operations

model_train.py  - Creation of PyTorchTrainer object for model training and hyperparameter tuning


misc/           - Folder containing miscellaneous functions for the project


data/           - Folder containing modules related to creating, filling,   updating and querying the sqlite3 database that contains the input dataset

    chargemol_analysis.py - contains functions for calculating bond order for each of the Materials Project structures

    database_management.py - contains functions for interfacing with the sqlite3 database

    MP_query -  contains functions for querying the MAterials Project and filling the database
