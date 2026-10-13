"""Sequence filtering and file conversion utilities."""

import os
import random
import numpy as np
import pandas as pd
from .base import write_seq


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
