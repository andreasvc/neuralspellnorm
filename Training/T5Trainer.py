from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from transformers import AdamWeightDecay
import tensorflow as tf
import random
from transformers import logging as hf_logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import textwrap
import argparse
import re
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
hf_logging.set_verbosity_error()

np.random.seed(1234)
tf.random.set_seed(1234)
random.seed(1234)


def create_arg_parser():
    '''Creating command line arguments'''
    parser = argparse.ArgumentParser()

    parser.add_argument("-tf", "--transformer", default="google/byt5-small",
                        type=str, help="this argument takes the pretrained "
                                       "language model URL from HuggingFace "
                                       "default is ByT5-small, please visit "
                                       "HuggingFace for full URL")
    parser.add_argument("-c_model", "--custom_model",
                        type=str, help="this argument takes a custom "
                                       "pretrained checkpoint")
    parser.add_argument("-train", "--train_data", default='training_data10k.txt',
                        type=str, help="this argument takes the train "
                                       "data file as input")
    parser.add_argument("-dev", "--dev_data", default='validation_data.txt', 
                        type=str, help="this argument takes the dev data file "
                        "as input")
    parser.add_argument("-lr", "--learn_rate", default=5e-5, type=float,
                        help="Set a custom learn rate for "
                             "the model, default is 5e-5")
    parser.add_argument("-bs", "--batch_size", default=8, type=int,
                        help="Set a custom batch size for "
                             "the pretrained language model, default is 8")
    parser.add_argument("-sl_train", "--sequence_length_train", default=155,
                        type=int, help="Set a custom maximum sequence length"
                                        "for the pretrained language model,"
                                        "default is 155")
    parser.add_argument("-sl_dev", "--sequence_length_dev", default=155,
                        type=int, help="Set a custom maximum sequence length"
                                        "for the pretrained language model,"
                                        "default is 155")
    parser.add_argument("-ep", "--epochs", default=1, type=int,
                        help="This argument selects the amount of epochs "
                             "to run the model with, default is 1 epoch")
    parser.add_argument("-es", "--early_stop", default="val_loss", type=str,
                        help="Set the value to monitor for earlystopping")
    parser.add_argument("-es_p", "--early_stop_patience", default=2,
                        type=int, help="Set the patience value for "
                                       "earlystopping, default is 2")
    args = parser.parse_args()
    return args


def read_data(data_file):
    '''Reading in data files'''
    with open(data_file) as file:
        data = file.readlines()

    text = []
    for d in data:
        text.append(d)
    return text


def create_data(data):
    '''Splitting Alpino format training data into separate 
    source and target sentences'''
    source_text = []
    target_text = []
    for x in data:
        source = []
        target = []
        spel = re.findall(r'\[.*?\]', x)
        if spel:
            for s in spel:
                s = s.split()
                if s[1] == '@alt':
                    target.append(''.join(s[2:3]))
                    source.append(''.join(s[3:-1]))
                elif s[1] == '@mwu_alt':
                    target.append(''.join(s[2:3]))
                    source.append(''.join(s[3:-1]).replace('-', ''))
                elif s[1] == '@mwu':
                    target.append(''.join(s[2:-1]))
                    source.append(' '.join(s[2:-1]))
                elif s[1] == '@postag':
                    target.append(''.join(s[-2]))
                    source.append(''.join(s[-2]))
                elif s[1] == '@phantom':
                    target.append(''.join(s[2]))
                    source.append('')

        target2 = []
        for t in target:
          if t[0] == '~':
            t = t.split('~')
            target2.append(t[1])
          else:
            target2.append(t)

        sent = re.sub(r'\[.*?\]', 'EMPTY', x)
        word_c = 0
        src = []
        trg = []
        for word in sent.split():
            if word == 'EMPTY':
                src.append(source[word_c])
                trg.append(target2[word_c])
                word_c += 1
            else:
                src.append(word)
                trg.append(word)
        source_text.append(' '.join(src))
        target_text.append(' '.join(trg))
    return source_text, target_text


def split_sent(data, max_length):
    '''Splitting sentences if longer than given max_length value'''
    short_sent = []
    long_sent = []
    for n in data:
        n = n.split('|')
        if len(n[1]) <= max_length:
            short_sent.append(n[1])
        elif len(n[1]) > max_length:
            n[1] = re.sub(r'(\s)+(?=[^[]*?\])', '$$', n[1])
            n[1] = n[1].replace("] [", "]##[")
            lines = textwrap.wrap(n[1], max_length, break_long_words=False)
            long_sent.append(lines)

    new_data = []
    for s in long_sent:
        for s1 in s:
            s1 = s1.replace(']##[', '] [')
            s1 = s1.replace('$$', ' ')
            s2 = s1.split()
            if len(s2) > 2:
                new_data.append(s1)

    for x in short_sent:
        new_data.append(x)
    return new_data


def preprocess_function(tk, s, t):
    '''tokenizing text and labels'''
    model_inputs = tk(s)

    with tk.as_target_tokenizer():
        labels = tk(t)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


def convert_tok(tok, sl):
    '''Convert tokenized object to Tensors and add padding'''
    input_ids = []
    attention_mask = []
    labels = []
    decoder_attention_mask = []
    for a, b, c, d in zip(tok['input_ids'], tok['attention_mask'], tok['labels'],
    tok['decoder_attention_mask']):
        input_ids.append(a)
        attention_mask.append(b)
        labels.append(c)
        decoder_attention_mask.append(d)

    input_ids_pad = pad_sequences(input_ids, padding='post', maxlen=sl)
    attention_mask_pad = pad_sequences(attention_mask, padding='post',
                                       maxlen=sl)
    labels_pad = pad_sequences(labels, padding='post', maxlen=sl)
    dec_attention_mask_pad = pad_sequences(decoder_attention_mask,
                                           padding='post', maxlen=sl)
    return {'input_ids': tf.constant(input_ids_pad), 'attention_mask':
        tf.constant(attention_mask_pad), 'labels': tf.constant(labels_pad),
        'decoder_attention_mask': tf.constant(dec_attention_mask_pad)}


def train_model(model_name, lr, bs, sl_train, sl_dev, ep, es, es_p, train, dev):
    '''Finetune and save a given T5 version with given parameters'''
    print('Training model: {}\nWith parameters:\nLearn rate: {}, '
          'Batch size: {}\nSequence length train: {}, sequence length dev: {}\n'
          'Epochs: {}'.format(model_name, lr, bs, sl_train, sl_dev, ep))

    tk = AutoTokenizer.from_pretrained(model_name)

    args = create_arg_parser()
    source_train, target_train = create_data(train)
    source_test, target_test = create_data(dev)

    if args.custom_model:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(args.custom_model,
                                                        from_pt=True)
    else:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_tok = preprocess_function(tk, source_train, target_train)
    dev_tok = preprocess_function(tk, source_test, target_test)

    tf_train = convert_tok(train_tok, sl_train)
    tf_dev = convert_tok(dev_tok, sl_dev)

    optim = AdamWeightDecay(learning_rate=lr)
    model.compile(optimizer=optim, loss=custom_loss,
                  metrics=[accuracy])
    ear_stop = tf.keras.callbacks.EarlyStopping(monitor=es, patience=es_p,
                                                restore_best_weights=True,
                                                mode="auto")
    model.fit(tf_train, validation_data=tf_dev, epochs=ep,
              batch_size=bs, callbacks=[ear_stop])
    model.save_weights('{}_weights.h5'.format(model_name[7:]))
    return model


def custom_loss(y_true, y_pred):
    '''Custom loss function'''
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def accuracy(y_true, y_pred):
    '''Custom accuracy function '''
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def main():
    args = create_arg_parser()

    lr = args.learn_rate
    bs = args.batch_size
    sl_train = args.sequence_length_train
    sl_dev = args.sequence_length_dev
    split_length_train = (sl_train - 5)
    split_length_dev = (sl_dev - 5)
    ep = args.epochs

    if args.transformer == 'google/flan-t5-small':
        model_name = 'google/flan-t5-small'
    elif args.transformer == 'google/byt5-small':
        model_name = 'google/byt5-small'
    elif args.transformer == 'google/mt5-small':
        model_name = 'google/mt5-small'
    else:
        model_name = 'Unknown'

    early_stop = args.early_stop
    patience = args.early_stop_patience

    train_d = read_data(args.train_data)
    dev_d = read_data(args.dev_data)
    train_data = split_sent(train_d, split_length_train)
    dev_data = split_sent(dev_d, split_length_dev)

    print('Train size: {}\nDev size: {}\n'.format(len(train_data),
                                                  len(dev_data)))
    print(train_model(model_name, lr, bs, sl_train, sl_dev,
                      ep, early_stop, patience, train_data, dev_data))


if __name__ == '__main__':
    main()
