import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import math
import warnings
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm

    
datasetDir = "/home/huangjinjun/workspace/minidataset"
dataset = load_dataset(path=datasetDir)

def preprocess_and_save(dataset, file_prefix, save_dir):
    global timestamp_set, timecnt, tokenized_dataset
    print("Process Starting-----")
    split_point = (int)(dataset.num_rows * 0.3)
    print(split_point)
    tgt_dataset = dataset.select(range(split_point))
    print(tgt_dataset)
    tgt_dataset.to_parquet('train-00000-of-00001.parquet')
preprocess_and_save(dataset['train'], 'train', datasetDir)