# %%
import os
import pandas as pd
from transformers import pipeline
import json

folder_contain_test_json = '/root/thesis/dataset/cosmos_anns_acm/acm_anns'
folder_contain_image = '/root/thesis/dataset'

# %%
import sys
# os.chdir('/root/thesis/acmmmcheapfakes/COSMOS')
sys.path.append('/root/thesis/acmmmcheapfakes/COSMOS')
# print(os.getcwd())
from COSMOS import evaluate_ooc

# %%
evaluate_ooc.DATA_DIR = folder_contain_image
evaluate_ooc.JSON_DIR = folder_contain_test_json

# %%
# evaluate_ooc.main(None)

# %%
cosmos_iou = pd.read_csv('pred_contexts.txt', header=None)
cosmos_iou.columns = ['iou']

# %%
sys.modules.pop('utils')
sys.modules.pop('utils.eval_utils')

# %%
os.chdir('/root/thesis/acmmmcheapfakes/OFA')
sys.path.remove('/root/thesis/acmmmcheapfakes/COSMOS')
sys.path.append('/root/thesis/acmmmcheapfakes/OFA')

# %%
# data = json.loads(open(os.path.join(folder_contain_test_json,'test_data.json')))
test_data = list(
    map(json.loads, open(os.path.join(folder_contain_test_json,'test_data.json')).readlines())
)
df = pd.DataFrame(test_data)

# %%
# sys.path.insert(0, '/root/thesis/acmmmcheapfakes/OFA/utils')
from OFA.main import run

# %%
result_ofa = run(df)

# %%
z = pd.read_csv('ofa_full.csv')

# %%
df = pd.concat([df, z[['c1_entail','c2_entail']]], axis=1)

# %%
# os.chdir('/root/thesis/acmmmcheapfakes')

# %%
from sbert import sbert

# %%
print('SBERT:')
print(sbert(['hello', 'hi']))

# %% [markdown]
# # Online

# %%
import os
import requests, json
import urllib.parse
from sentence_transformers import SentenceTransformer, util
from multiprocessing.pool import ThreadPool
from PIL import Image
import requests
from io import BytesIO
import torch
import pycurl
import certifi
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm


# %%
from online_search import retrieve

# %%
# tqdm.pandas()
df['caption1_online_check'] = df.progress_apply(lambda x: retrieve(os.path.join(folder_contain_image,x.img_local_path),x.caption1), axis=1) 
df['caption2_online_check'] = df.progress_apply(lambda x: retrieve(os.path.join(folder_contain_image,x.img_local_path),x.caption2), axis=1) 

# %% [markdown]
# # NLI


# %%
import torch
# device = `torch.device('cuda',1)a`
classifier = pipeline("text-classification", model = "microsoft/deberta-xlarge-mnli",
device=torch.device('cuda',1))

# %%
df['nli_label'] = df.progress_apply(lambda x: classifier(x.caption1+x.caption2)[0], axis=1)
df['nli_label_reverse'] = df.progress_apply(lambda x: classifier(x.caption2+x.caption1)[0], axis=1)

# %%
print(classifier(
    """
    Vladimir Putin said that 'ISIS will regret' assassinating a Russian ambassador since he is 'not Obama' and this is 'not Benghazi.' is fake.A quote ostensibly uttered by Russian president Vladimir Putin criticizing President Obama is a fabricated one.
    """
)[0]['label'])

# %% [markdown]
# # BERT

# %%



