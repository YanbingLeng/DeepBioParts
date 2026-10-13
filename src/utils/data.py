"""Sequence encoding, I/O, and k-mer utilities for model usage."""

import os
import numpy as np
import pandas as pd
import torch
import collections
import matplotlib.pyplot as plt


def open_fa(file):
    record = []
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                record.append(line.rstrip('\n'))
    return record


def convert(n, x):
    list_a = [0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
    list_b = []
    while True:
        s,y = divmod(n,x)
        list_b.append(y)
        if s == 0:
            break
        n = s
    list_b.reverse()
    res = []
    for i in range(x):
        res.append(0)
    res0 = []
    for i in list_b:
        res0.append(list_a[i])
    for i in range(len(res0)):
        res[x - i - 1] = res0[len(res0) - i - 1]
    return res


def kmer_frequency(valid_path, k=4, save_path='cache/', save_name='99'):
    print('Start saving the frequency figure......')
    bg = ['T', 'C', 'G', 'A']
    valid_kmer, ref_kmer = collections.OrderedDict(), collections.OrderedDict()
    kmer_name = []
    for i in range(4**k):
        nameJ = ''
        cov = convert(i, 4)
        for j in range(k):
                nameJ += bg[cov[j]]
        kmer_name.append(nameJ)
        valid_kmer[nameJ], ref_kmer[nameJ] = 0, 0
    valid_df = pd.read_csv(valid_path)
    generated_seq = list(valid_df['output'])
    input_seq = list(valid_df['input'])
    valid_num, ref_num = 0, 0
    for i in range(len(generated_seq)):
        generated_seq[i] = generated_seq[i].strip()
        for j in range(len(generated_seq[i]) - k + 1):
            k_mer = generated_seq[i][j : j + k]
            if k_mer in valid_kmer:
                valid_kmer[k_mer] += 1
                valid_num += 1
           
    for i in range(len(input_seq)):
        input_seq[i] = input_seq[i].strip()
        for j in range(len(input_seq[i]) - k + 1):
            k_mer = input_seq[i][j : j + k]
            if k_mer in ref_kmer:
                ref_kmer[k_mer] += 1
                ref_num += 1

    for i in kmer_name:
        # Avoid division by zero
        if ref_num > 0 and valid_num > 0:
            ref_kmer[i], valid_kmer[i] = ref_kmer[i]/ref_num, valid_kmer[i]/valid_num
        elif ref_num > 0:
            ref_kmer[i] = ref_kmer[i]/ref_num
            valid_kmer[i] = 0
        elif valid_num > 0:
            valid_kmer[i] = valid_kmer[i]/valid_num
            ref_kmer[i] = 0
        else:
            ref_kmer[i], valid_kmer[i] = 0, 0
    plt.plot(list(ref_kmer.values()))
    plt.plot(list(valid_kmer.values()))
    plt.legend(['real distribution', 'model distribution'])
    plt.title('{}_mer frequency'.format(k))
    plt.xlabel('{}_mer index'.format(k))
    plt.ylabel('{}_mer frequency'.format(k))
    plt.savefig('{}{}_{}_mer_frequency.png'.format(save_path, save_name, k))
    plt.close()
    print('Saving end!')


def load_seq_data(datafile):
    """Load unsupervised sequence data.

    Args:
        datafile: Path to the data file.

    Returns:
        oh: One-hot encoded sequences.
        seqs: Raw sequences.
    """
    seqs = open_fa(datafile)
    oh = seq2onehot(seqs)
    
    return oh, seqs


def seq2onehot(seqs):
        """One-hot encode variable-length sequences.

        Args:
            seqs: List of sequences (may differ in length).

        Returns:
            padded_onehot: Padded one-hot array of shape (num_seqs, max_len, 4).
        """
        module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        # Handle empty input
        if not seqs:
            return np.zeros((0, 0, 4), dtype=np.float32)

        max_len = max(len(s) for s in seqs)

        # Initialize result array
        padded_onehot = np.zeros((len(seqs), max_len, 4), dtype=np.float32)

        for i, seq in enumerate(seqs):
            seq_len = len(seq)
            for j, base in enumerate(seq):
                base = base.upper()
                if base == 'A':
                    padded_onehot[i,j] = module[3]
                elif base == 'C':
                    padded_onehot[i,j] = module[1]
                elif base == 'G':
                    padded_onehot[i,j] = module[2]
                elif base in ('T', 'U'):
                    padded_onehot[i,j] = module[0]
                else:  # N and other unknown characters
                    padded_onehot[i,j] = [0,0,0,0]

            # Sequences shorter than max_len are zero-padded
            if seq_len < max_len:
                padded_onehot[i, seq_len:] = [0,0,0,0]

        return padded_onehot


def onehot2seq(onehot_data):
    """Convert a one-hot matrix back into nucleotide sequences.

    Args:
        onehot_data (numpy.ndarray or torch.Tensor): One-hot matrix of shape
            (num_samples, seq_len, 4).

    Returns:
        list: List of nucleotide sequences.
    """
    # Convert PyTorch tensors to numpy arrays
    if 'torch' in str(type(onehot_data)):
        onehot_data = onehot_data.cpu().numpy()

    # Mapping from one-hot index to nucleotide
    mapping = {0: 'T', 1: 'C', 2: 'G', 3: 'A'}
    sequences = []

    for sample in onehot_data:
        seq = ""
        for position in sample:
            # Find the index of the value-1 entry
            index = np.argmax(position)
            if np.sum(position) == 0:  # [0, 0, 0, 0] indicates an unknown base
                seq += 'N'
            else:
                seq += mapping[index]
        sequences.append(seq)

    return sequences


def process_template(template, generated_seq):
    """Replace X placeholders in a template with a generated sequence.

    Args:
        template: DNA sequence template containing 'X' placeholders.
        generated_seq: Generated DNA sequence to fill in.

    Returns:
        str: The completed sequence.
    """
    if 'X' not in template:
        return template

    x_count = template.count('X')

    if len(generated_seq) < x_count:
        raise ValueError(f"Generated sequence length ({len(generated_seq)}) is insufficient "
                         f"to replace {x_count} X placeholders in the template")

    result = ""
    gen_idx = 0

    for char in template:
        if char == 'X':
            result += generated_seq[gen_idx]
            gen_idx += 1
        else:
            result += char

    return result


def save_sequence(output_tensor, input_tensor, save_path='', name=''):
    """Save validation-set input and output tensors to a CSV file.

    Args:
        output_tensor: Output tensor of shape [batch, seq_len, features].
        input_tensor: Input tensor of shape [batch, seq_len, features].
        save_path: Output directory (default '').
        name: CSV filename prefix (default '').

    Returns:
        str: Full path to the CSV file.
    """
    # Ensure CPU tensors
    if torch.is_tensor(output_tensor):
        output_tensor = output_tensor.cpu().detach()
    if torch.is_tensor(input_tensor):
        input_tensor = input_tensor.cpu().detach()

    # Convert tensors to DNA sequences
    output_seqs = onehot2seq(output_tensor)
    input_seqs = onehot2seq(input_tensor)

    # Build DataFrame
    df = pd.DataFrame({
        'output': output_seqs,
        'input': input_seqs
    })

    os.makedirs(save_path, exist_ok=True)

    csv_path = os.path.join(save_path, f'{name}sequences.csv')

    df.to_csv(csv_path, index=False)

    print(f"Sequences saved to: {csv_path}")

    return csv_path


def compute_kmer_vector(sequence, k=3):
    """Compute the k-mer frequency vector for a sequence.

    Args:
        sequence: DNA sequence string.
        k: k-mer length (default 3).

    Returns:
        k-mer frequency vector (numpy array).
    """
    import itertools

    # All possible k-mer combinations
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    # Initialize k-mer count vector
    kmer_vector = np.zeros(len(all_kmers), dtype=np.float32)

    # Count k-mers
    sequence = sequence.upper()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if 'N' not in kmer:  # Skip k-mers containing unknown bases
            if kmer in kmer_to_idx:
                kmer_vector[kmer_to_idx[kmer]] += 1

    # Normalize
    total_kmers = np.sum(kmer_vector)
    if total_kmers > 0:
        kmer_vector = kmer_vector / total_kmers

    return kmer_vector
