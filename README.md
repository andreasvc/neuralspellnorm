# Master Thesis

Repository containing all code and data used for master thesis: 
* * Historical Dutch Spelling Normalization with Pre-trained Language Models * *.

## The repository contains the following directories:

### Evaluation

-This directory contains the python notebook and file used to evaluate the predictions for each trained model. 
The different subdirectories each contain three .txt files, these are the predictions for that specific model.

### Gold-Data

-This directory contains the gold data for validation and testing in .txt format.
The * * annotation_guidelines.pdf * * file contains the guidelines for annotating or correcting 19th century Dutch novels.

### Silver-Data

-This directory contains the individual novels as .txt file for the 5k and 10k finetuning datasets.
The * * training_dat10k.txt * * and * * training_data5k.txt * * are the novels combined and cleaned.

### Training

-This directory contains all code used for finetuning the different T5 versions. 
Where * * T5Trainer * * and * * T5Predict * * are used to train and test the models.
