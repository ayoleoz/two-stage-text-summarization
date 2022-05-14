import argparse
from datetime import datetime
import os
import time
import numpy as np
import json

# WarmupLinearSchedule doesn't work, see: https://github.com/huggingface/transformers/issues/2082
# from transformers import GPT2LMHeadModel,AdamW, WarmupLinearSchedule 
from transformers import GPT2LMHeadModel,AdamW, get_linear_schedule_with_warmup 
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.notebook import trange, tqdm
from IPython.display import clear_output
from datasets import load_dataset, load_metric # Huggingface
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

from utils import add_special_tokens, set_seed, sample_seq

from torch.utils.data import Dataset
import json

#please change default arguments if needed

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
parser.add_argument("--seed", default=42, type=int,  help="seed to replicate results")
parser.add_argument("--n_gpu", default=1, type=int,  help="no of gpu available")
parser.add_argument("--gradient_accumulation_steps", default=32, type=int, help="gradient_accumulation_steps")
parser.add_argument("--batch_size", default=1, type=int,  help="batch_size")
parser.add_argument("--num_workers", default=4, type=int,  help="num of cpus available")
parser.add_argument("--device", default=torch.device('cuda'), help="torch.device object")
parser.add_argument("--num_train_epochs", default=5, type=int,  help="no of epochs of training")
parser.add_argument("--output_dir", default='./output', type=str,  help="path to save evaluation results")
parser.add_argument("--model_dir", default='./weights', type=str,  help="path to save trained model")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
parser.add_argument("--root_dir", default='./CNN/gpt2_1024_data', type=str, help="location of json dataset.")
parser.add_argument("--ids_file", default='./CNN/ids.json', type=str, help="location of train, valid and test file indexes")
args = parser.parse_args([])
print(args)

def write_json(i, article, abstract, cnndm_id, is_cut=0, path=''):
	""" Saves a json file.
    Args:
        - id (int): order number
        - article (tensor): tokenized article 
        - abstract (tensor): tokenized abstract
        - is_cut (boolean): the tokenized article is cut by 1024 or not
                            0 - cut; 1 - original
        - cnndm_id (string): the id from cnndm dataset
    """

	file = f"{path}/{i}.json"
	js_example = {}
	js_example['id'] = i
	js_example['article'] = article
	js_example['abstract'] = abstract
	js_example['is_cut'] = is_cut
	js_example['cnndm_id'] = cnndm_id
	with open(file, 'w') as f:
		json.dump(js_example, f, ensure_ascii=False)

class GPT21024CNNDM(Dataset):
    '''Load the data tokenized by GPT2 in Summarization task 
       - The data is concatenated as {article, abstract}, 
         and truncated by the limitation of 1024 token size
    '''
    def __init__(self, root_dir, mode='train',length=None):
        self.root_dir = root_dir
        self.tokenizer = add_special_tokens()
        # self.idxs = os.listdir(f'{root_dir}/{mode}') # cool; but slow
        self.mode = mode
        # self.len = len(self.idxs) if length == None else length
        if length != None:
            self.len = length
        elif mode == 'train':
            self.len = 287113
        elif mode == 'validation':
            self.len = 13368 
        elif mode == 'test':
            self.len = 11490

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        file_name = os.path.join(self.root_dir, self.mode, f"{idx}.json")
        with open(file_name,'r') as f:
              data = json.load(f)
        text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'article': text, 'sum_idx': len(data['article'])}
        return sample