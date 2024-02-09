# Historical Dutch Spelling Normalization with Pre-trained Language Models

## This repository contains the following directories:

### Evaluation

This directory contains the Python notebook (`evaluation.ipynb`) and file (`evaluate_functions.py`) used to evaluate the predictions for each trained model. 
The different subdirectories contain three .txt files; these are the predictions for that specific model on the three test novels.

### Gold-Data

This directory contains the separate gold novels used for the validation and test set.
The `Annotation_Guidelines_19th_Dutch_Spelling.pdf` file contains the guidelines for annotating or correcting 19th-century Dutch novels.

### Silver-Data

This directory contains the automatically annotated novels as .txt files for the 5k and 10k finetuning datasets.
The `training_data10k.txt` and `training_data5k.txt` are the novels combined and cleaned.
The `literature_list.txt` contains the names of all the novels taken from Project Gutenberg.

### Training

This directory contains the code used for finetuning and testing the different T5 versions. 
Where `T5Trainer` and `T5Predict` are used to train and test the models.

__Example:__
To run the `T5Trainer` script for finetuning ByT5 with a custom set of parameters, use the command:
```
python3 T5Trainer.py -tf google/byt5-small -train training_data.txt -dev validation_data.txt -lr 5e-5 -bs 32 -sl_train 155 -sl_dev 455 -ep 20 -es val_accuracy -es_p 3
```
See `T5Trainer` for a full explanation of each of the arguments.

To run the `T5Predict` with a trained ByT5 model on the three gold novels, use the following command:
```
python3 T5Predict.py -tf 'google/byt5-small' -weights 'byt5-small_weights.h5' -temp 0.4 -cs 40 -n_beam 2 -test1 'Multatuli_MaxHavelaar.txt' -test2 'ConanDoyle_SherlockHolmesDeAgraSchat.txt' -test3 'Nescio_Titaantjes.txt'
```
See `T5Predict` for a full explanation of each of the arguments.

`Gutenberg_Data.ipynb` is the notebook used to extract and download novels from Project Gutenberg.
This notebook uses the [Dutch Literature Pipeline](https://github.com/andreasvc/dutchlitpreproc) repository from van Cranenburgh.  

### Pretraining

This directory contains the code used to further pretrain ByT5.
The `pretrain_jax.py` is the Python file to pretrain the model, which uses custom (hyper)parameters; see [run_t5_mlm_flax.py](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py) for further explanation.

__Example:__
To run the `pretrain_jax.py` script for pretraining ByT5, use the following command:
```
python3 pretrain_jax.py \
        --output_dir "." \
        --model_name_or_path "google/byt5-small" \
        --model_type "t5" \
        --config_name "google/byt5-small" \
        --tokenizer_name "google/byt5-small" \
        --do_train --do_eval \
        --adafactor \
        --train_file "pretraining_train.txt" \
        --validation_file "pretraining_dev.txt" \
        --max_seq_length "512" \
        --per_device_train_batch_size "8" \
        --per_device_eval_batch_size "8" \
        --learning_rate "0.0005" \
        --overwrite_output_dir \
        --num_train_epochs "25" \
        --logging_steps "10000" \
        --save_steps "50000" \
        --eval_steps "10000" \
        --weight_decay "0.001" \
        --warmup_steps "10000" \
        --mean_noise_span_length "20" \
        --grad_acc "1"
```

the `train_file` and `validation_file` arguments are where the training and validation files are given as input, which can be created with the `create_data.ipynb` notebook.

The `create_data.ipynb` notebook is used to convert Dutch data into a single train and validation file.
The `convert_jax_to_torch.py` is the script used to convert the pretrained weights in Flax format into a Pytorch format.

### DownstreamEval

Scripts to run the evaluation on downstream tasks. See the README for more information.

