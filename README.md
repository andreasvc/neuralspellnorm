# Master Thesis

Repository containing all code and data used for master thesis: 
__Historical Dutch Spelling Normalization with Pre-trained Language Models__

## The repository contains the following directories:

### Evaluation

This directory contains the python notebook and file used to evaluate the predictions for each trained model. 
The different subdirectories each contain three .txt files, these are the predictions for that specific model on the three test novels.

### Gold-Data

This directory contains the gold data for validation and testing in .txt format.
The _annotation_guidelines.pdf_ file contains the guidelines for annotating or correcting 19th century Dutch novels.

### Silver-Data

This directory contains the individual novels as .txt file for the 5k and 10k finetuning datasets.
The _training_data10k.txt_ and _training_data5k.txt_ are the novels combined and cleaned.

### Training

This directory contains all python code used for finetuning and testing the different T5 versions. 
Where _T5Trainer_ and _T5Predict_ are used to train and test the models.
