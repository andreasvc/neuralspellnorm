from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import random
from transformers import logging as hf_logging
import numpy as np
from tqdm import tqdm
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
    '''Create command line arguments'''
    parser = argparse.ArgumentParser()

    parser.add_argument("-tf", "--transformer", default="google/byt5-small",
                        type=str, help="Language model type")
    parser.add_argument("-weights", "--model_weights", type=str,
                        help="Trained model weights (Tensorflow format)")
    parser.add_argument("-temp", "--temperature", type=float, default=1.0,
                        help="Temperature weight for generation")
    parser.add_argument("-cs", "--chunk_size", type=int, default=40,
                        help="Batch size for test set")
    parser.add_argument("-n_beam", "--num_beams", type=int, default=2,
                        help="beam search number")
    parser.add_argument("-test1", "--test_set1", type=str,
                        help="Test set to let the model predict on")
    parser.add_argument("-test2", "--test_set2", type=str,
                        help="Test set to let the model predict on")
    parser.add_argument("-test3", "--test_set3", type=str,
                        help="Test set to let the model predict on")
    args = parser.parse_args()
    return args


def read_data(data_file):
    '''Read in .txt file'''
    with open(data_file, 'r') as file:
        data = file.readlines()

    text = []
    for d in data:
        text.append(d)
    return text


def save_data(data_file, file_name):
    '''Save predictions as .txt file'''
    file_name = file_name + '_pred.txt'
    with open(file_name, 'w') as file:
        for line in data_file:
            file.write(line + '\n')


def create_data(data):
    '''Splitting Alpino format data into source and target sentences'''
    source_text = []
    target_text = []
    for d in data:
        d = d.split('|')
        x = d[1].strip()
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


def batch(iterable, n=1):
    '''Returns given list file into smaller batches'''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def model_predict(model, tok, novel, chunk_size, temp, n_beams):
    '''Predict with given model and (hyper)parameters on a .txt file'''
    x_test, _ = create_data(novel)
    predictions = []
    for test in tqdm(batch(x_test, chunk_size)):
        tokenized = tok(test, padding=True, return_tensors='tf')
        pred = model.generate(input_ids=tokenized['input_ids'],
                              attention_mask=tokenized['attention_mask'],
                              max_new_tokens=500, temperature=temp,
                              num_beams=n_beams)
        for p in pred:
            predictions.append(
                tok.decode(p, text_target=True, skip_special_tokens=True))
    return predictions


def main():
    args = create_arg_parser()

    model_name = args.transformer
    weights = args.model_weights
    temp = args.temperature
    chunk_size = args.chunk_size
    n_beams = args.num_beams

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.load_weights(weights)

    if args.test_set1:
        print('Predicting on {}'.format(args.test_set1))
        data1 = read_data(args.test_set1)
        pred1 = model_predict(model, tokenizer, data1, chunk_size, temp, n_beams)
        save_data(pred1, args.test_set1[:-4])

    if args.test_set2:
        print('Predicting on {}'.format(args.test_set2))
        data2 = read_data(args.test_set2)
        pred2 = model_predict(model, tokenizer, data2, chunk_size, temp, n_beams)
        save_data(pred2, args.test_set2[:-4])

    if args.test_set3:
        print('Predicting on {}'.format(args.test_set3))
        data3 = read_data(args.test_set3)
        pred3 = model_predict(model, tokenizer, data3, chunk_size, temp, n_beams)
        save_data(pred3, args.test_set3[:-4])


if __name__ == '__main__':
    main()
