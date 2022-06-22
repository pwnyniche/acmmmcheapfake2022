import os
import torch
import numpy as np
from utils.eval_utils import eval_snli_ve
from data.mm_data.snli_ve_dataset import collate
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import pandas as pd
from tqdm import tqdm
from tasks.mm_tasks.snli_ve import SnliVeTask

# from models.ofa import OFAModel
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)
IMAGE_DATA_DIR = os.getenv('IMAGE_DATA_DIR')

# specify some options for evaluation
parser = options.get_generation_parser()
input_args = ["", 
"--task=snli_ve",
"--path=checkpoints/snli_ve_large_best.pt",
"--bpe-dir=utils/BPE", 
"--batch-size=8 "]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# task = tasks.setup_task(cfg.task)
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths(cfg.common_eval.path)
)

# Image transform
from torchvision import transforms
from utils.trie import Trie
from data import data_utils

import re
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
eos_idx = task.src_dict.eos()
tgt_dict = task.tgt_dict
max_src_length = 80

# device='cuda'

def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip())) 
      if not word.startswith('<code_') and not word.startswith('<bin_') else word
      for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

def pre_caption(caption, max_words):
    caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def construct_sample(uniq_id ,image, hypothesis, caption,label):
    if label == 'contradiction':
        label = 'no'
    elif label == 'entailment':
        label = 'yes'
    elif label == 'neutral':
        label = 'maybe'
    else:
        raise NotImplementedError

    patch_image = patch_resize_transform(image)
    patch_mask = torch.tensor([True])

    hypothesis = pre_caption(hypothesis, max_src_length)
    src_item = encode_text(' does the image describe " {} "?'.format(hypothesis))
    tgt_item = encode_text(" {}".format(label))
    ref_dict = {label: 1.0}

    # caption = pre_caption(caption, max_src_length)
    # src_item = encode_text(' can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))

    src_item = torch.cat([bos_item, src_item, eos_item])

    prev_output_item = torch.cat([src_item[:-1], tgt_item])
    target_item = torch.cat([prev_output_item[1:], eos_item])
    decoder_prompt = src_item[:-1]
    target_item[:-len(tgt_item)-1] = 1 # tgt_dict.pad()

    constraint_trie = Trie(tgt_dict.eos())
    constraint_mask = torch.zeros((len(target_item), len(tgt_dict))).bool()
    start_idx = len(target_item) - len(tgt_item) - 1
    for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
        constraint_prefix_token = [tgt_dict.bos()] + target_item[start_idx:i].tolist()
        constraint_nodes = constraint_trie.get_next_layer(constraint_prefix_token)
        constraint_mask[i][constraint_nodes] = True
    example = {
            "id": uniq_id,
            "source": src_item.to(device),#
            "patch_image": patch_image.to(device),#
            "patch_mask": patch_mask.to(device),#
            "target": target_item.to(device),
            "prev_output_tokens": prev_output_item.to(device),
            "decoder_prompt": decoder_prompt.to(device),
            "ref_dict": ref_dict,
            "constraint_mask": constraint_mask.to(device)
        }
    return example

# device = 'cuda'
models[0].to(device);

def create_sample(row):
    with torch.no_grad():
        d1 = construct_sample(row["img_local_path"]+'__c1',
            Image.open(os.path.join(IMAGE_DATA_DIR,row["img_local_path"])),
            row['caption1'], '','entailment')
        d2 = construct_sample(row["img_local_path"]+'__c2',
            Image.open(os.path.join(IMAGE_DATA_DIR,row["img_local_path"])), 
            row['caption2'], '','entailment')
        sample = collate([d1,d2], pad_idx, eos_idx)
        result = eval_snli_ve(task,None,models,sample)
    row['c1_entail'] = result[0][0]['e_score'].numpy().tolist()
    row['c2_entail'] = result[0][1]['e_score'].numpy().tolist()
    return row

# torch.manual_seed(0)
# with torch.no_grad():
#     d0 = construct_sample("I1",Image.open('/root/thesis/dataset/public_test_mmsys/661.jpg'), 
#     'The Red Cross burial workers carry a box containing the body of an 11-month-old girl', '','entailment')
#     sample = collate([d0,d0,d0], pad_idx, eos_idx)
#     r = eval_snli_ve(task,None,models,sample)
# print(r)

def run (df):
    # df = pd.read_csv(data_path)
    torch.manual_seed(0)
    tqdm.pandas()
    z=df.progress_apply(lambda x: create_sample(x), axis=1)
    z.to_csv('ofa_full.csv', index=False)
    # l = []
    # for g in z:
    #     l.append(g[0])
    #     l.append(g[1])
    # split_l = np.array_split(l,200)
    # c1=[]
    # c2=[]

    # for l_ in tqdm(split_l):
    #     with torch.no_grad():
    #         sample = collate(l_, pad_idx, eos_idx)
    #         result = eval_snli_ve(task,None,models,sample)

    #     for r in result[0]:
    #         img_local_path,cap = r['uniq_id'].split('__')
    #         if cap=='c1':
    #             c1.append(r['e_score'].numpy().tolist())
    #         else:
    #             c2.append(r['e_score'].numpy().tolist())
    # return (c1,c2)
