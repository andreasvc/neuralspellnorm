#!/usr/bin/env python
import re
from tqdm import tqdm
import textwrap
from sklearn.metrics import precision_score, recall_score
import shutil
import datasets


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


def cal_err(raw, gold, pred):
    '''Calculates the error reduction rath, script taken from Rob van der Goot'''
    cor = 0
    changed = 0
    total = 0

    if len(gold) != len(pred):
        return 'Error: gold normalization contains a different numer of sentences(' + str(len(gold)) + ') compared to system output(' + str(len(pred)) + ')'
       
    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            return 'Error: a sentence has a different length in you output, check the order of the sentences'
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            if wordRaw != wordGold:
                changed += 1
            if wordGold == wordPred:
                cor += 1
            total += 1

    accuracy = float(cor) / total
    lai = float(total - changed) / total
    err = float(accuracy - lai) / (1-lai)
    return 'Baseline Accuracy: {:.2f}%\nAccuracy: {:.2f}%\nError Reduction Rate: {:.2f}%'.format((lai * 100), (accuracy * 100), (err * 100)) 


def cal_pre_rec(gold, pred):
    '''Calculate precision and recall for a whole .txt file'''
    gold_novel = []
    pred_novel = []
 
    for sentGold, sentPred in zip(gold, pred):
        if len(sentGold) != len(sentPred):
            return 'Error: gold normalization contains a different numer of sentences(' + str(len(gold)) + ') compared to system output(' + str(len(pred)) + ')'
        
    for sentGold, sentPred in zip(gold, pred):
        for wordGold, wordPred in zip(sentGold, sentPred):
            gold_novel.append(wordGold)
            pred_novel.append(wordPred)
                
    precision = precision_score(gold_novel, pred_novel, average='macro', zero_division=True)
    recall = recall_score(gold_novel, pred_novel, average='macro', zero_division=True)
    return 'Avg Precision: {:.2f}%\nAvg Recall: {:.2f}%'.format((precision * 100), (recall * 100))


def cal_chrf(gold, pred, word_order, model_type, chrf_type):
    '''Calculate ChrF and ChrF++ between prediction and gold sentences'''
    chrf = datasets.load_metric('chrf')
    nested_gold = []
    for g in gold:
        nested_gold.append([g])

    results = chrf.compute(predictions=pred, references=nested_gold, 
                           word_order=word_order)
    print('{}: {}'.format(chrf_type, model_type))
    scores = []
    for k, v in results.items():
        scores.append(v)
        print('   {}: {}'.format(k, round(v, 2)))
        
        
def cal_chrf_split(gold, pred, word_order, model_type, chrf_type):
    '''Calculate ChrF and ChrF++ between prediction and gold sentences,
    sentences are splitted on word level'''
    chrf = datasets.load_metric('chrf')
    nested_gold = []
    join_pred = []
    for g, p in zip(gold, pred):
        g = ' '.join(g)
        p = ' '.join(p)
        nested_gold.append([g])
        join_pred.append(p)

    results = chrf.compute(predictions=join_pred, references=nested_gold, 
                           word_order=word_order)
    print('{}: {}'.format(chrf_type, model_type))
    scores = []
    for k, v in results.items():
        scores.append(v)
        print('   {}: {}'.format(k, round(v, 2)))


def align_gold(gold):
    '''Align gold sentences'''
    gold = ' '.join(gold).strip()
    combin_words_gold = [('aan te spreken', 'aan@te@spreken'), ('op aan', 'op@aan'),
                         ('op te merken', 'op@te@merken'), ('om mijnentwille', 'om@mijnentwille'),
                         ('God weet waarheen', 'God@weet@waarheen'), ('van wie het', 'van@wie@het'),
                         ('van mijn', 'van@mijn'), ('in plaats', 'in@plaats'), ('rozenbomen hout', 'rozenbomen@hout'),
                         (",'s", ", 's"), (" krant ' estaan ", " krant'estaan "), ('XIII .', 'XIII.'),
                         ('XII .', 'XII.'), ('XI .', 'XI.'), (' toe juichen ', ' toe@juichen '), (' zo -iets ', ' zo@-iets ')]
    for word_pair1 in combin_words_gold:
        if word_pair1[0] in gold:
            gold = gold.replace(word_pair1[0], word_pair1[1])

    adj_gold = []
    for g in gold.split():
        g = g.replace('@', ' ')
        adj_gold.append(g)
    return adj_gold


def align_pred(pred):
    '''Align prediction sentences'''
    pred = ' '.join(pred).strip()
    combin_words_pred = [(' der ', ' der - '), (' des ', ' des - '), ('S .', 'S.'), ('A .', 'A.'), 
                         ('D .', 'D.'), ('P .', 'P.'), ('Z .', 'Z.'), ('V .', 'V.'), ('I .', 'I.'),
                         ('b . v .', 'b.v.'), ('W .', 'W.'), ('N .', 'N.'), ('J .', 'J.'),
                         ('Mrs .', 'Mrs.'), ('H .', 'H.'), ('Dr .', 'Dr.'), ('St .', 'St.'),
                         ('3 , 37', '3,37'), ('X .', 'X.'), ('G .', 'G.'), (' zooiemand ', ' zoo iemand '),
                         ('enz .', 'enz.'), ('No .', 'No.'), ('Mr .', 'Mr.')]
    for word_pair2 in combin_words_pred:
        if word_pair2[0] in pred:
            pred = pred.replace(word_pair2[0], word_pair2[1])

    adj_pred = []
    for p in pred.split():
        p = p.replace('@', ' ')
        adj_pred.append(p)
    return adj_pred


def align_silver(pred):
    '''Align rule-based sentences'''
    pred = ' '.join(pred).strip()
    combin_words_pred = [(' om mijnentwille ', ' om@mijnentwille '), (' aan te spreken ', ' aan@te@spreken '), 
                         (' zo -iets ', ' zo@-iets '), (' op te merken ', ' op@te@merken '), 
                         (' toe juichen ', ' toe@juichen '), (' er tegen ', ' er@tegen '), (' te zamen ', ' te@zamen '), 
                         (' rozenbomen hout ', ' rozenboom@hout '), (' te kort ', ' te@kort '), 
                         (' honderd veertig ', ' honderd@veertig '), ('Bij voorbeeld ', 'Bij@voorbeeld '),
                         (' een zelfde ', ' een@zelfde '), (" als 't u belieft ", " als@'t@u@belieft "),
                         (' ver gezocht ', ' ver@gezocht '), (' er voor ', ' er@voor '), (' mij zelf ', ' mij@zelf '), 
                         (' God weet waarheen ', ' God@weet@waarheen '), (' mijner ', ' mijner - '), 
                         (' der ', ' der - '), ('Vier en dertig ', 'Vier@en@dertig '), (' des ', ' des - '),
                         (' van wie het ', ' van@wie@het '), (' in plaats ', ' in@plaats '), (' op aan ', ' op@aan '),
                         (' te voren ', ' te@voren ')]
    for word_pair2 in combin_words_pred:
        if word_pair2[0] in pred:
            pred = pred.replace(word_pair2[0], word_pair2[1])

    adj_pred = []
    for p in pred.split():
        p = p.replace('@', ' ')
        adj_pred.append(p)
    return adj_pred


def evaluate_T5(gold_data, predictions, model_type, verbose=False):
    '''Evaluate T5 models for the four metrics'''
    gold = []
    pred = []
    raw = []
    source, target = create_data(gold_data)
    for p, g, r in zip(predictions, target, source):
        p = p.strip()
        g = g.strip()
        r = r.strip()
        p = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', p)
        p = p.replace('. . .', '...').replace('... .', '....').replace('. .', '..')
        g = g.replace(" 's", "'s").replace(" 'm", "'m").replace('... .', '....')
        r = r.replace(" 's", "'s").replace(" 'm", "'m").replace('... .', '....')
            
        p = p.split()
        g = g.split()
        r = r.split()
    
        if len(g) != len(r):
            g = align_gold(g)
            r = align_pred(r)
        
        if len(g) < 1:
            p = ''
    
        if len(g) > len(p):
            g = align_gold(g)

        if len(p) > len(g):
            p = align_pred(p)
       
        if len(p) == len(g) and len(r) == len(g):
            gold.append(g)
            pred.append(p)
            raw.append(r)
        else:
            p = align_pred(p)
            g = align_gold(g)
            r = align_gold(r)
            if len(p) != len(g):
                if verbose:
                    print(len(r), r)
                    print(len(g), g)
                    print(len(p), p)
                    print('\n')
                gold.append(g)
                pred.append(r)
                raw.append(r)
            else:
                gold.append(g)
                pred.append(p)
                raw.append(r)
                                    
    print(cal_err(raw, gold, pred))
    print(cal_pre_rec(gold, pred))
    
    print('\n')
    print('ChrF scores with original predictions:')
    cal_chrf(target, predictions, 2, model_type, 'ChrF++')
    cal_chrf(target, predictions, 0, model_type, 'ChrF')
    
    print('\n')
    print('ChrF scores with aligned predictions:')
    cal_chrf_split(gold, pred, 2, model_type, 'ChrF++')
    cal_chrf_split(gold, pred, 0, model_type, 'ChrF')
    return pred, gold, raw


def evaluate_rulebased(gold_data, predictions, model_type, verbose=False):
    '''Evaluate Rule-based system for the four metrics'''
    mis_gold = []
    mis_pred = []
    gold = []
    pred = []
    raw = []
    source, target = create_data(gold_data)
    for p, g, r in zip(predictions, target, source):
        p = p.strip().split()
        g = g.strip().split()
        r = r.strip().split()
        
        if len(p) != len(g) or len(g) != len(r):
            p = align_silver(p)
            g = align_silver(g)
            r = align_silver(r)
        
        if len(p) == len(g) and len(g) == len(r):
            gold.append(g)
            pred.append(p)
            raw.append(r)
        else:
            if verbose:
                print(len(p), p)
                print(len(g), g)
                print(len(r), r)
                print('\n')

            
    print(cal_err(raw, gold, pred))
    print(cal_pre_rec(gold, pred))
    
    print('ChrF scores with original predictions:')
    cal_chrf(target, predictions, 2, model_type, 'ChrF++')
    cal_chrf(target, predictions, 0, model_type, 'ChrF')
    
    print('ChrF scores with aligned predictions:')
    cal_chrf_split(gold, pred, 2, model_type, 'ChrF++')
    cal_chrf_split(gold, pred, 0, model_type, 'ChrF')
    return pred, gold, raw
    

if __name__ == '__main__':
    main()
    