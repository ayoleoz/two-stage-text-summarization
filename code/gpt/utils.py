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


from dataset import GPT21024Dataset 
from utils import add_special_tokens, set_seed, sample_seq

def compute_perplexity(args, model, eval_dataset, ignore_index, global_step=None):
    """ Returns perplexity score on validation dataset.
		Args:
			args: dict that contains all the necessary information passed by user while training
			model: finetuned gpt/gpt2 model
			eval_dataset: GPT21024Dataset object for validation data
			global_step: no. of times gradients have backpropagated
			ignore_index: token not considered in loss calculation
	"""
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = torch.tensor(batch['article']).to(args.device), torch.tensor(batch['article']).to(args.device)
        with torch.no_grad():
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    
    return perplexity
    
    
def postprocess_text(preds, labels):
    """Postprocess output to prepare for ROUGE evaluation."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLsum expects newline after each sentence
    preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ['\n'.join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def score_rouge(preds, refs):
    """Compute ROUGE scores on a list of predicted summaries given reference summaries."""
    metric = load_metric('rouge')
    preds, refs = postprocess_text(preds, refs)
    result = metric.compute(predictions=preds, references=refs, use_stemmer=True)

    # Extract only mid scores from ROUGE
    # Note that rougeLsum should be used as the ROUGE-L score
    result = {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}
    return result
    
def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    
    context = torch.tensor(context, dtype=torch.long).to(device)
    # context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        for _ in range(length):
        # for _ in tqdm(range(length)):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            # add by yifei: idk why new model will sometime out of 1024 so patch here temporary
            # print(generated.shape[1])
            if generated.shape[1] >= 1024:
                break
    return generated
    
def generate_single_sample(sample, tokenizer, model, length=100, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda'), return_result=False):
    """ Generate summaries for "num" number of articles.
        Args:
            data = one sample from GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            model = gpt/gpt2 model
            num = number of articles for which summaries has to be generated
            eval_step = can be True/False, checks generating during evaluation or not
    """
    idx = sample['sum_idx']
    context = sample['article'][:idx].tolist()
    summary = sample['article'][idx+1:][:100].tolist()
    generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p)
    generated_text = generated_text[0, len(context):].tolist()
    pred = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
    pred = tokenizer.convert_tokens_to_string(pred)
    # article = tokenizer.decode(context)
    ref = tokenizer.decode(summary, skip_special_tokens=True)
    ref = ''.join(ref.strip().split('\n')) 

    if return_result:
        return ref, pred 
        # return article, ref, pred 
    else:
        # print(f'Article:\n{article}\n')
        print(f'Actual_summary:\n{ref}\n')
        print(f'Generated_summary:\n{pred}\n')