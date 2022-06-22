# TODO: COSMOS, OFA, SBERT, NLI, CLIP device
import sys
import torch
import pandas as pd
from transformers import pipeline
import json
import os
import numpy as np
from tqdm import tqdm
from sbert import sbert
from online_search import retrieve
tqdm.pandas()
classifier = pipeline("text-classification", model = "microsoft/deberta-xlarge-mnli",
device=torch.device('cuda',0))

BASE_DIR = os.getenv('BASE_DIR')
ANNOTATION_DATA_DIR = os.getenv('ANNOTATION_DATA_DIR')
IMAGE_DATA_DIR = os.getenv('IMAGE_DATA_DIR')
# '/root/thesis/dataset/cosmos_anns_acm/acm_anns'
# '/root/thesis/dataset'

test_data = list(
    map(json.loads, open(os.path.join(ANNOTATION_DATA_DIR,'test_data.json')).readlines())
)
df = pd.DataFrame(test_data)
df['bert_base_score'] = df['bert_base_score'].astype(float)

def print_div(s):
    term_size = os.get_terminal_size()
    print('=' * term_size.columns)
    print('||' + s)
    print('=' * term_size.columns)

sys.path.append(os.path.join(BASE_DIR, 'COSMOS'))

from COSMOS import evaluate_ooc

print_div('Running COSMOS')
evaluate_ooc.main(None)
sys.modules.pop('utils')
sys.modules.pop('utils.eval_utils')


print_div('Running OFA')
os.chdir(os.path.join(BASE_DIR, 'OFA'))
sys.path.remove(os.path.join(BASE_DIR, 'COSMOS'))
sys.path.append(os.path.join(BASE_DIR, 'OFA'))

from OFA.main import run
run(df)

cosmos_iou = pd.read_csv(os.path.join(BASE_DIR, 'pred_contexts.txt'), header=None)
cosmos_iou.columns = ['iou']
df = pd.concat([df, cosmos_iou['iou']], axis=1)
ofa_result = pd.read_csv('ofa_full.csv')
df = pd.concat([df, ofa_result[['c1_entail','c2_entail']]], axis=1)

os.chdir(BASE_DIR)

print_div('Running NLI')
df['nli_label'] = df.progress_apply(lambda x: classifier(x.caption1+x.caption2)[0], axis=1)
df['nli_label_reverse'] = df.progress_apply(lambda x: classifier(x.caption2+x.caption1)[0], axis=1)
def nli(x):
    if (x.nli_label['label'] == 'CONTRADICTION' and x.nli_label_reverse['label'] == 'CONTRADICTION'):
        return 'CONTRADICTION'
    if (x.nli_label['label'] == 'ENTAILMENT' or x.nli_label_reverse['label'] == 'ENTAILMENT'):
        if (x.nli_label['label'] != 'CONTRADICTION' and x.nli_label_reverse['label'] != 'CONTRADICTION'):
            return 'ENTAILMENT' 
    return 'NEUTRAL'
df['nli'] = df.progress_apply(lambda x:nli(x), axis=1) 


print_div('Running Misleading')
keywords = "fake, hoax, fabrication, supposedly, falsification, propaganda, deflection, deception, contradicted, defamation, lie, misleading, deceive, fraud, concocted, bluffing, made up, double meaning, alternative facts, trick, half-truth, untruth, falsehoods, inaccurate, disinformation, misconception"
df['cap1_mis']=df.progress_apply(lambda x: sbert([x.caption1_modified,keywords]),axis=1)
df['cap2_mis']=df.progress_apply(lambda x: sbert([x.caption2_modified,keywords]),axis=1)
def get_fake_scores(row):
    new_scores = [0,0,0]
    if row['cap1_mis'] > row['cap2_mis'] and row['cap1_mis']>0.15:
        c_fake = row['caption2'].rstrip('.') + ' is not genuine.'
        new_scores[0] = sbert([c_fake,row['caption1']])

        c_fake = row['caption2'].rstrip('.') + ' is fake.'
        new_scores[1] = sbert([c_fake,row['caption1']])

        c_fake = row['caption2'].rstrip('.') + ' wasn\'t true.'
        new_scores[2] = sbert([c_fake,row['caption1']])
    elif row['cap1_mis'] < row['cap2_mis'] and row['cap2_mis']>0.15: 
        c_fake = row['caption1'].rstrip('.') + ' is not genuine.'
        new_scores[0] = sbert([c_fake,row['caption2']])

        c_fake = row['caption1'].rstrip('.') + ' is fake.'
        new_scores[1] = sbert([c_fake,row['caption2']])

        c_fake = row['caption1'].rstrip('.') + ' wasn\'t true.'
        new_scores[2] = sbert([c_fake,row['caption2']])
    return new_scores
df['false_scores'] = df.progress_apply(lambda x: get_fake_scores(x), axis=1)


def predict_everything_online(row):
    if row.nli == 'CONTRADICTION':
        return (True,'NLI')
    if row.nli == 'ENTAILMENT':
        return (False,'NLI')
    false_scores = row['false_scores']
    if np.any(np.array(false_scores) >  row['bert_base_score']+0.05):
        return (True, 'MIS')
    if row['iou']>0.25:
        if row['bert_base_score']<0.5:
            c1 = retrieve(os.path.join(IMAGE_DATA_DIR,row['img_local_path']),row['caption1'])[0]
            c2 = retrieve(os.path.join(IMAGE_DATA_DIR,row['img_local_path']),row['caption2'])[0]
            c1 = c1 > 0.97
            c2 = c2 > 0.97
            
            if c1 and c2:
                return (False, 'ONL')

            if (c1):
                if row['c2_entail']>0.25:
                    return (False, 'MAT')
            if (c2):
                if row['c1_entail']>0.25:
                    return (False, 'MAT')

            return (True, 'BERT')
        return (False, 'BERT')
    return (False,'COSMOS')

def evaluate(df, func):
    df['result'] =  df.progress_apply(lambda x:func(x), axis=1)
    df['predict'] =  df['result'].apply(lambda x:x[0])
    df['method'] =  df['result'].apply(lambda x:x[1])
    confusion_matrix = pd.crosstab(df['predict'], df['context_label'], rownames=['Predicted'], colnames=['Actual'])
    print(confusion_matrix)
    result = (confusion_matrix[0][0]+confusion_matrix[1][1])/len(df)
    print('Accuracy:', result)
    method_acc = df.groupby('method').apply(lambda g: \
        ((g['context_label']==g['predict']).sum() / len(g),len(g) ))
    print(method_acc.head(10))

print_div('Evaluating...')
evaluate(df, predict_everything_online)
 
df.to_csv('result_df.csv', index=False)
df['predict'].to_csv('predict.csv', index=False)