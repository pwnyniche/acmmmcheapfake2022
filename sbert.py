from __future__ import absolute_import, division, unicode_literals
import sys
import io
import numpy as np
import logging
import argparse
import torch
import random
# from transformers import *
from transformers import AutoConfig, AutoTokenizer,AutoModelWithLMHead
import bert_utils
import json
from tqdm import tqdm

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
class Args:
  def __init__(args, batch_size, max_seq_length, seed, model_type, embed_method, context_window_size, layer_start, tasks, device):
    args.batch_size = batch_size
    args.max_seq_length = max_seq_length
    args.seed = seed
    args.model_type = model_type
    args.embed_method = embed_method
    args.context_window_size = context_window_size
    args.layer_start = layer_start
    args.tasks = tasks
    args.device = device
args = Args(64,128,42,"binwang/bert-base-nli-stsb","ave_last_hidden",2,4,'sts', 2)
# -----------------------------------------------
# Set device
# torch.cuda.set_device(2)
# device = torch.device("cuda", 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# -----------------------------------------------
# Set seed
set_seed(args)
# Set up logger
# logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)
# Set Model
params = vars(args)

config = AutoConfig.from_pretrained(params["model_type"], cache_dir="./cache")
config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained(params["model_type"], cache_dir="./cache")
model = AutoModelWithLMHead.from_pretrained(
    params["model_type"], config=config, cache_dir="./cache"
)

model.to(params["device"]);

def sbert(sentences):
    # -----------------------------------------------
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > params["max_seq_length"]:
            sent_ids = sent_ids[: params["max_seq_length"]]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = params["max_seq_length"] - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == params["max_seq_length"]
        assert len(sent_mask) == params["max_seq_length"]

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

    features_mask = np.array(features_mask)

    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    model.zero_grad()

    with torch.no_grad():
        features = model(**inputs)[1]

    # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
    # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
    all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

    embed_method = bert_utils.generate_embedding(params["embed_method"], features_mask)
    embedding = embed_method.embed(params, all_layer_embedding)
    similarity = (
        embedding[0].dot(embedding[1])
        / np.linalg.norm(embedding[0])
        / np.linalg.norm(embedding[1])
    )
    return similarity