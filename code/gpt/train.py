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
from utils import * 
from dataset import *

def train(args, train_data, valid_data):
    # load pretrained GPT2
    tokenizer = add_special_tokens()
    ignore_index = tokenizer.pad_token_id
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    # writer = SummaryWriter('./logs')
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)
    
    sample = valid_data[0] 
    loss_train_stor, perplex_val_stor = [], []
    COMPUTE_PERPLEXITY = True
    
    for epoch in tqdm(range(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Step")):
            inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            model.train()
            logits = model(inputs)[0]
            idx = batch['sum_idx'].item() # index of separator token
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss /= args.gradient_accumulation_steps
            loss.backward()
            loss_train_stor.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                # writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
                # logging_loss = tr_loss
                print(f"Loss: {loss.item():.2e}")
                
                if (step + 1) % (10*args.gradient_accumulation_steps) == 0:
                    if COMPUTE_PERPLEXITY:
                        perplexity = compute_perplexity(args, model, valid_data, ignore_index, global_step)
                        print(f'Perplexity (validation): {perplexity}')
                        perplex_val_stor.append(perplexity)
                    print(f'After {global_step+1} updates:\n')
                    generate_single_sample(sample, tokenizer, model, device=args.device)
        
        # Save model, loss and perplexity
        model_path = f'{checkpoint_folder}/{model_name}-epoch-{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
    
        np.save(f'{checkpoint_folder}/{model_name}-train-loss-epoch-{epoch}.npy', np.array(loss_train_stor))
        if COMPUTE_PERPLEXITY:
            np.save(f'{checkpoint_folder}/{model_name}-train-perplex-epoch-{epoch}.npy', np.array(perplex_val_stor))
    
        if epoch < args.num_train_epochs - 1:
            clear_output()