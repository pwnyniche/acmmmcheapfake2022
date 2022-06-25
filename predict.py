import os
import numpy as np
from online_search import retrieve

IMAGE_DATA_DIR = os.getenv('IMAGE_DATA_DIR')

def predict_baseline(row):
    if row['iou']>0.5:
        return (row['bert_base_score']<0.5, 'BERT')
    return [False,'COSMOS']

def predict_baseline_025(row):
    if row['iou']>0.25:
        return (row['bert_base_score']<0.5, 'BERT')
    return [False,'COSMOS']

def predict_nli(row):
    if row.nli == 'CONTRADICTION':
        return [True,'NLI']
    if row.nli == 'ENTAILMENT':
        return [False,'NLI']
    if row['iou']>0.25:
        return (row['bert_base_score']<0.5, 'BERT')
    return [False,'COSMOS']

def predict_fabricate(row):
    false_scores = row['false_scores']
    if np.any(np.array(false_scores) >  row['bert_base_score']+0.05):
        return [True, 'MIS']
    if row['iou']>0.25:
        return (row['bert_base_score']<0.5, 'BERT')
    return [False,'COSMOS']

def predict_online_match(row):
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
    return [False,'COSMOS']

def predict_final(row):
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