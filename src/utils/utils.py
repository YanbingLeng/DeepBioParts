import sys
import os
import numpy as np
import pandas as pd
from .base import *
import os, sys
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

def filter_for_experiment(seqpath, gc_strength=[0.2,0.8], poly_strength=5):
    
    seqs = open_fa(seqpath)
    res1, res2 = [], []
    
    # Step1: filter GC
    min_gc_ratio = gc_strength[0]
    max_gc_ratio = gc_strength[1]
    
    gc_res = []
    for string in seqs:
        g_count = string.count('G')  
        c_count = string.count('C')  
        
        total_count = len(string)
        gc_ratio = (g_count + c_count) / total_count
        gc_res.append(gc_ratio)
        
        if gc_ratio >= min_gc_ratio and gc_ratio <= max_gc_ratio:  
            res1.append(string)
    
    # Step2: filter poly A,T
    
    poly_res = []
    for string in seqs:
        count_A = string.count('A' * poly_strength)  
        count_T = string.count('T' * poly_strength)  
        
        poly_res.append(count_A + count_T)
        
        if count_A == 0 and count_T == 0:  
            res2.append(string)  

    seqpath_txt = os.path.splitext(seqpath)[0] + "_filter.txt"
    seqpath_csv = os.path.splitext(seqpath)[0] + "_filter.csv"
    
    res_dict = {"seqs": seqs, "gc content":gc_res, "poly at": poly_res}
    df = pd.DataFrame(res_dict)
    df.to_csv(seqpath_csv)

    res = list( set(res1)  & set(res2) )
    write_seq(seqpath_txt, res)
    
    print("We have conducted quality assessment for each sequence you provided and selected the sequences that meet your screening strength.")
    print("Results have been save separately saved in .csv and .txt file with _filter suffix")
    return

def csv2fasta(csv_path, data_path, data_name):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realB = list(results['realB'])
    f2 = open(data_path + data_name + '_realB.fasta','w')
    j = 0
    for i in realB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()
    f2 = open(data_path + data_name + '_fakeB.fasta','w')
    j = 0
    for i in fakeB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class EarlyStopping_P:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, stop_order='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None      # stores the raw best score
        self.early_stop = False
        self.stop_order = stop_order
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric=None, model=None, val_loss=None):
        # Backward-compatible: if val_loss is provided, use it as the metric
        if val_loss is not None:
            val_metric = val_loss
        # Decide whether the metric improved
        if self.best_score is None:
            # First call
            self.best_score = val_metric
            self.save_checkpoint(val_metric, model)
        else:
            # Check improvement based on stop_order
            if self.stop_order == 'max':
                improved = val_metric > self.best_score + self.delta
            else:  # 'min'
                improved = val_metric < self.best_score - self.delta

            if improved:
                # Improved: reset the counter and save the model
                self.best_score = val_metric
                self.save_checkpoint(val_metric, model)
                self.counter = 0
            else:
                # Not improved: increment the counter
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        if self.verbose:
            self.trace_func(f'Metric improved --> {val_metric:.6f}. Saving model ...')
        torch.save(model.state_dict(), self.path)


class EarlyStopping_G:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', stop_order='max',
                 min_js_divergence=0.01, min_acc_improvement=0.001, trace_func=print, monitor='both'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.stop_order = stop_order
        self.monitor = monitor
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if self.stop_order == 'max':
            self.val_metric = -np.inf
        else:
            self.val_metric = np.inf
        self.trace_func = trace_func
        # Unified initialization of all metrics
        if self.stop_order == 'max':
            self.val_loss_min = -np.inf
            self.best_score = -np.inf
        else:
            self.val_loss_min = np.inf
            self.best_score = np.inf
        self.best_js = float('inf')
        self.best_acc = 0.0

        # Ensure the save directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, metric=None, model=None, current_js=None, current_acc=None, val_loss=None):
        # Backward-compatible: if val_loss is provided, use it as the metric
        if val_loss is not None:
            metric = val_loss

        # Initialize best metrics
        if not hasattr(self, 'best_js'):
            self.best_js = float('inf')
        if not hasattr(self, 'best_acc'):
            self.best_acc = 0.0

        # Print current metrics in real time
        print(f"\nEarlyStopping Progress:")
        print(f"Current metric: {metric:.4f} (Best: {self.best_score:.4f})")
        js_str = f"{current_js:.4f}" if current_js is not None else "N/A"
        print(f"Current JS divergence: {js_str} (Best: {self.best_js:.4f})")
        acc_str = f"{current_acc:.4f}" if current_acc is not None else "N/A"
        print(f"Current Accuracy: {acc_str} (Best: {self.best_acc:.4f})")
        print(f"Patience counter: {self.counter}/{self.patience}")

        # Check the JS-divergence condition
        js_condition = False
        if current_js is not None:
            if current_js < self.best_js - self.delta:
                self.best_js = current_js
                js_condition = True
            else:
                js_condition = False

        # Check the accuracy condition
        acc_condition = False
        if current_acc is not None:
            if current_acc > self.best_acc + self.delta:
                self.best_acc = current_acc
                acc_condition = True
            else:
                acc_condition = False

        # Check the reconstruction-loss condition
        recon_condition = False
        if self.monitor == 'reconstruction_loss':
            if self.stop_order == 'min':
                if self.best_score is None or metric < self.best_score - self.delta:
                    recon_condition = True
            elif self.stop_order == 'max':
                if self.best_score is None or metric > self.best_score + self.delta:
                    recon_condition = True

        # Decide which condition to use based on the monitor argument
        if self.monitor == 'js_divergence':
            condition = js_condition
        elif self.monitor == 'accuracy':
            condition = acc_condition
        elif self.monitor == 'reconstruction_loss':
            condition = recon_condition
        else:  # 'both'
            condition = js_condition or acc_condition or recon_condition

        # Update the early-stopping state
        if condition:
            # Metric improved
            if self.best_score is None:
                self.best_score = metric
            else:
                self.best_score = metric
            self.counter = 0
            if model is not None:
                self.save_checkpoint(metric, model)
            print("Checkpoint saved with improved metrics:")
            if current_js is not None and current_acc is not None:
                print(f"JS: {current_js:.4f}, Acc: {current_acc:.4f}")
            elif current_js is not None:
                print(f"JS: {current_js:.4f}, Acc: N/A")
            elif current_acc is not None:
                print(f"JS: N/A, Acc: {current_acc:.4f}")
            else:
                print("Metrics updated (no JS/Acc provided)")
        else:
            # Metric did not improve
            self.counter += 1
            print(f"Early stopping counter increased to {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered due to:")
                if self.monitor == 'js_divergence' or self.monitor == 'both':
                    if not js_condition:
                        print("- JS divergence not improving")
                if self.monitor == 'accuracy' or self.monitor == 'both':
                    if not acc_condition:
                        print("- Accuracy not improving")
                if self.monitor == 'reconstruction_loss' or self.monitor == 'both':
                    if not recon_condition:
                        print("- Reconstruction loss not improving")

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease. Returns True if saved successfully.'''
        if self.verbose:
            self.trace_func(f'Updation changed ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        try:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
            return True
        except Exception as e:
            self.trace_func(f'Failed to save checkpoint: {str(e)}')
            return False


def read_fa(file_name):
    seq = []
    with open(file_name, 'r') as f:
        for item in f:
            if '>' not in item:
                seq.append(item.strip('\n'))
    return seq


def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def open_exp(file, operator = 'log2'):
    record = []
    f = open(file,'r')
    for item in f:
        record.append(float(item))
    max_num = max(record)
    min_num = min(record)
    result = []
    for item in record:
        if operator == 'log2':
            result.append(np.log2(item))
        elif operator == 'zero-one':
            result.append((item - min_num) / (max_num - min_num))
        else:
            result.append(item)
    f.close()
    return result

def write_exp(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(str( np.round(data[i], 2)) + '\n')
        i = i + 1
    f.close()
    return

def dataset_shuffle(seqpath, exppath, savetag=False): 
    seqs = open_fa(seqpath)
    expr = open_exp(exppath, "direct")
    idx = np.arange(len(seqs))
    random.shuffle(idx)
    
    seqs = np.array(seqs)[idx]
    expr = np.array(expr)[idx]
    
    if savetag:
        seqpath_new = os.path.splitext(seqpath)[0] + "_shuffle.txt"
        exppath_new = os.path.splitext(exppath)[0] + "_shuffle.txt"
        write_seq(seqpath_new, seqs)
        write_exp(exppath_new, expr)
        print("The new shuffled file has been stored with _shuffle suffix\n")
    
    return seqs, expr

def dataset_split(seqpath, exppath, ratio=0.8, savetag=False):
    seqs = open_fa(seqpath)
    expr = open_exp(exppath, "direct")
    
    total_length = len(seqs)
    r = int(total_length * ratio)
    
    seqs_train = seqs[0:r]
    expr_train = expr[0:r]
    seqs_test = seqs[r:total_length]
    expr_test = expr[r:total_length]
    
    if savetag:
        seqpath_train_new = os.path.splitext(seqpath)[0] + "_train.txt"
        exppath_train_new = os.path.splitext(exppath)[0] + "_train.txt"
        seqpath_test_new = os.path.splitext(seqpath)[0] + "_test.txt"
        exppath_test_new = os.path.splitext(exppath)[0] + "_test.txt"

        write_seq(seqpath_train_new, seqs_train)
        write_exp(exppath_train_new, expr_train)
        write_seq(seqpath_test_new, seqs_test)
        write_exp(exppath_test_new, expr_test)
        print("The new shuffled file has been stored with _train and _test suffix\n")
    
    return seqs_train, seqs_test, expr_train, expr_test