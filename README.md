# CS230_Project
Project Goal:
Almost every chemical we interact with requires a catalyst to create. Today, the activity of catalysts is determined computationally using Density Functional Theory (DFT). Unfortunately, this method is extremely expensive and can be a research bottleneck.

The convolution operation has proven to be an excellent feature extractor in the world of computer vision. This one operation is crucial to a wide variety of image-processing tasks, such as image classification and segmentation. We plan to create an analogous operation that can be applied to surface science. This operation will enable machine learning algorithms to expedite catalyst design. The fundamental mapping DFT provides, which we wish to learn, is one from a chemical structure to its energy.

Guide to files:

build_dataset.py   - functions for building the datasets

evalauate.py       - functions for evaluating models

train.py           - functions for training models

synthesize_results.py - functions for analyzing many models performance

model/             - folder containing functions for the neural net models
    data_loader.py    - functions for making and fetching pytorch dataloaders

    modules.py        - custom pytorch.nn Modules

    net.py            - file containing the actual model


misc/           - Folder containing miscellaneous functions for the project


data/           - Folder containing modules related to creating, filling,   updating and querying the sqlite3 database that contains the input dataset

    chargemol_analysis.py - contains functions for calculating bond order for each of the Materials Project structures

    database_management.py - contains functions for interfacing with the sqlite3 database

    MP_query -  contains functions for querying the MAterials Project and filling the database

experiments/    - Folder containing the different experiments run on the model
