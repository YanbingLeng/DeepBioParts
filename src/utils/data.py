'''
data helper
charmap: {0: 'T', 1: 'C', 2: 'G', 3: 'A'}
'''
import os, sys
# Dynamically add the project root to sys.path so `from src.xxx import` works from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
import collections
import time
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

def load_supervise_data(supervise_file, batch_size, batch_dict_name = ['data_train', 'label_train','data_valid','label_valid']):
    """Load supervised learning data.

    Args:
        supervise_file: Path to the supervised data file.
        batch_size: Batch size.
        batch_dict_name: Dictionary keys for the batch data.

    Returns:
        SupervisedData object.
    """
    data = SupervisedData(supervise_file, batch_size, batch_dict_name=batch_dict_name)
    return data

def split_data(data, r=0.9, max_val_samples=10000):
    """Split a dataset into training and validation sets.

    Args:
        data: Data to split.
        r: Training set fraction (default 0.9).
        max_val_samples: Maximum number of validation samples (default 10000).

    Returns:
        data_train: Training set data.
        data_val: Validation set data.
    """
    total_samples = data.shape[0]

    # Validation set size by ratio
    val_size_by_ratio = int(total_samples * (1 - r))

    # Cap validation set at max_val_samples
    val_size = min(val_size_by_ratio, max_val_samples)

    # Recompute training set size if the validation set was capped
    train_size = total_samples - val_size

    # Shuffle indices
    idx = np.random.permutation(total_samples)

    # Split into training and validation
    idx_train, idx_val = idx[:train_size], idx[train_size:train_size+val_size]
    data_train, data_val = data[idx_train], data[idx_val]

    print(f"Dataset split complete: total={total_samples}, train={train_size}, val={val_size}")
    print(f"train:val ratio = {train_size/val_size:.2f}:1")
    
    return data_train, data_val

def split_data_supervised(onehot, label, r=0.9):
    idx = np.random.permutation(onehot.shape[0])
    n = int(onehot.shape[0] * r)
    idx_train, idx_val = idx[:n], idx[n:]
    onehot_train, onehot_val = onehot[idx_train], onehot[idx_val]
    label_train, label_val = label[idx_train], label[idx_val]
    return onehot_train, label_train, onehot_val, label_val

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


'''
supervised_data class, for the semi-supervised training process
'''
class SupervisedData():
    def __init__(self, supervise_file, batch_size=64, batch_dict_name=None, shuffle=True):
        self._shuffle = shuffle
        self._batch_dict_name = batch_dict_name

        self._batch_size = batch_size
        self._load_files(supervise_file)
        self._seq_id_train = 0
        self._seq_id_test = 0
    
    
    def _load_files(self, supervise_file):
        
        supervise_file_path = supervise_file
        supervise_data = pd.read_csv(supervise_file_path)
        
        seqs_list, label_list = [], []
        
        
        # New CSV format: first column is the sequence, second is the label.
        # Assumes column names are "sequence" and "label".
        seq_column = supervise_data.columns[0]  # First column: sequence
        label_column = supervise_data.columns[1]  # Second column: label
        
        for i in range(len(supervise_data)):
            seq = supervise_data.loc[i, seq_column]
            label = supervise_data.loc[i, label_column]
            
            seqs_list.append(seq)
            label_list.append(label)
        
        onehot_list = seq2onehot(seqs_list)
        
        self.onehot_list = np.array(onehot_list)
        self.label_list = np.array(label_list)
        self.seqs_list = np.array(seqs_list)
        
        self.onehot_list_train, self.label_list_train, self.onehot_list_test, self.label_list_test = split_data_supervised(self.onehot_list, self.label_list, 0.9)
        self.datanum_train=self.onehot_list_train.shape[0] # (8153, 165, 4), (8153, )
        self.datanum_test=self.onehot_list_test.shape[0] # (906, 165, 4), (906, )
        self._shuffle_files(labeluse=True)

    def size(self,labeluse=False):
        return self.onehot_list.shape[0]
    
    def _shuffle_files(self,train_test_flag='train',labeluse=False):
        if self._shuffle:
            if train_test_flag=='train':
                idxs = np.arange(self.datanum_train)
                np.random.shuffle(idxs)
                self.onehot_list_train = self.onehot_list_train[idxs]
                if labeluse:
                    self.label_list_train = self.label_list_train[idxs]
            else:
                idxs = np.arange(self.datanum_test)
                np.random.shuffle(idxs)
                self.onehot_list_test = self.onehot_list_test[idxs]
                if labeluse:
                    self.label_list_test = self.label_list_test[idxs]

    def next_batch_dict(self, labeluse=False):  # Convert next_batch output list into a dict
        batch_data = self.next_batch(labeluse=labeluse)
        data_dict = {key: data for key, data in zip(self._batch_dict_name, batch_data)}
        return data_dict
    
    def next_batch(self,labeluse=False):  # Generate training and validation batches
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        
        #train
        start_train = self._seq_id_train
        self._seq_id_train += self._batch_size
        end_train = self._seq_id_train
        batch_data_train = self.onehot_list_train[start_train:end_train]
        batch_label_train = self.label_list_train[start_train:end_train]
        
        self._shuffle_files(train_test_flag='train',labeluse=labeluse)
        if self._seq_id_train + self._batch_size > self.datanum_train:
            self._seq_id_train = 0
        
        #test
        start_test = self._seq_id_test
        self._seq_id_test += self._batch_size
        end_test = self._seq_id_test
        batch_data_test = self.onehot_list_test[start_test:end_test]
        batch_label_test = self.label_list_test[start_test:end_test]

        self._shuffle_files(train_test_flag='test',labeluse=labeluse)
        if self._seq_id_test + self._batch_size > self.datanum_test:
            self._seq_id_test = 0

        return [batch_data_train, batch_label_train,batch_data_test, batch_label_test]
    
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


def cluster_based_split(seqs, labels, test_size=0.2, similarity_threshold=0.8,
                        kmer_size=3, random_state=42, verbose=True):
    """Cluster-based dataset split that preserves inter-set sequence divergence.

    Procedure:
        1. Vectorize sequences by k-mer frequency.
        2. Compute pairwise cosine similarity.
        3. Group similar sequences by hierarchical clustering.
        4. Split by whole clusters so sequences in the same cluster never
           straddle the train/test boundary.

    Args:
        seqs: List of sequences.
        labels: Corresponding labels.
        test_size: Test set fraction (default 0.2).
        similarity_threshold: Similarity threshold for hierarchical clustering
            (default 0.8). Higher values produce more, smaller clusters
            (stricter split); lower values produce fewer, larger clusters
            (looser split).
        kmer_size: k-mer length (default 3).
        random_state: Random seed (default 42).
        verbose: Whether to print detailed progress (default True).

    Returns:
        train_seqs, train_labels, test_seqs, test_labels
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity

    np.random.seed(random_state)

    if verbose:
        print("=" * 70)
        print("Similarity-based dataset split")
        print("=" * 70)
        print(f"Total samples: {len(seqs)}")
        print(f"Test set fraction: {test_size}")
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"k-mer length: {kmer_size}")

    # Step 1: compute k-mer vectors for all sequences
    if verbose:
        print("\n[1/5] Computing k-mer frequency vectors...")

    kmer_vectors = np.array([compute_kmer_vector(seq, kmer_size) for seq in seqs])

    if verbose:
        print(f"  k-mer vector dim: {kmer_vectors.shape[1]}")

    # Step 2: pairwise cosine similarity matrix
    if verbose:
        print("\n[2/5] Computing similarity matrix...")

    similarity_matrix = cosine_similarity(kmer_vectors)

    # Convert to distance matrix (distance = 1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Ensure non-negative (avoid floating-point precision issues)
    distance_matrix = np.maximum(distance_matrix, 0)

    # Ensure exact-zero diagonal (avoid floating-point precision issues)
    np.fill_diagonal(distance_matrix, 0)

    if verbose:
        print(f"  Similarity stats: min={np.min(similarity_matrix):.4f}, "
              f"max={np.max(similarity_matrix):.4f}, "
              f"mean={np.mean(similarity_matrix):.4f}")

    # Step 3: hierarchical clustering
    if verbose:
        print("\n[3/5] Running hierarchical clustering...")

    Z = linkage(squareform(distance_matrix), method='average')

    # Cut into clusters by a distance threshold (1 - similarity_threshold)
    max_distance = 1.0 - similarity_threshold
    cluster_labels = fcluster(Z, t=max_distance, criterion='distance')

    n_clusters = len(np.unique(cluster_labels))
    if verbose:
        print(f"  Clustering complete: {n_clusters} clusters identified")
        print(f"  Cluster size distribution:")
        cluster_sizes = [np.sum(cluster_labels == c) for c in np.unique(cluster_labels)]
        print(f"    Smallest cluster: {min(cluster_sizes)} sequences")
        print(f"    Largest cluster: {max(cluster_sizes)} sequences")
        print(f"    Mean cluster size: {np.mean(cluster_sizes):.1f} sequences")

    # Step 4: split by whole clusters
    if verbose:
        print("\n[4/5] Splitting dataset by clusters...")

    unique_clusters = np.unique(cluster_labels)
    np.random.shuffle(unique_clusters)  # Randomly order clusters

    n_test_clusters = int(len(unique_clusters) * test_size)

    test_clusters = unique_clusters[:n_test_clusters]
    train_clusters = unique_clusters[n_test_clusters:]

    train_indices = np.where([label in train_clusters for label in cluster_labels])[0]
    test_indices = np.where([label in test_clusters for label in cluster_labels])[0]

    train_seqs = [seqs[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_seqs = [seqs[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Step 5: validate split quality
    if verbose:
        print("\n[5/5] Validating split quality...")
        print(f"  Train size: {len(train_seqs)} ({len(train_seqs)/len(seqs)*100:.1f}%)")
        print(f"  Test size: {len(test_seqs)} ({len(test_seqs)/len(seqs)*100:.1f}%)")

        # Average similarity between train and test
        train_kmers = kmer_vectors[train_indices]
        test_kmers = kmer_vectors[test_indices]
        cross_similarity = cosine_similarity(train_kmers, test_kmers)

        print(f"  Train-test mean similarity: {np.mean(cross_similarity):.4f}")
        print(f"  Train-test max similarity: {np.max(cross_similarity):.4f}")

        # Check for highly similar cross-set sequence pairs
        highly_similar_pairs = np.argwhere(cross_similarity > similarity_threshold)
        if len(highly_similar_pairs) > 0:
            print(f"  Warning: found {len(highly_similar_pairs)} highly similar cross-set sequence pairs!")
        else:
            print(f"  OK: no cross-set sequence pairs with similarity > {similarity_threshold}")

        print("=" * 70)
        print("Dataset split complete!")
        print("=" * 70)

    return train_seqs, train_labels, test_seqs, test_labels


class ClusterBasedKFold:
    """Cluster-based K-fold cross-validation by sequence similarity.

    Procedure:
        1. Vectorize sequences by k-mer frequency.
        2. Compute pairwise cosine similarity.
        3. Group similar sequences by hierarchical clustering.
        4. Split by whole clusters so sequences in the same cluster never
           straddle the train/validation boundary.

    Args:
        n_splits: Number of folds (default 5).
        similarity_threshold: Similarity threshold for hierarchical clustering
            (default 0.8). Higher values produce more, smaller clusters
            (stricter split); lower values produce fewer, larger clusters
            (looser split).
        kmer_size: k-mer length (default 3).
        random_state: Random seed (default 42).
        verbose: Whether to print detailed progress (default True).
    """
    def __init__(self, n_splits=5, similarity_threshold=0.8, kmer_size=3,
                 random_state=42, verbose=True):
        self.n_splits = n_splits
        self.similarity_threshold = similarity_threshold
        self.kmer_size = kmer_size
        self.random_state = random_state
        self.verbose = verbose

        # Populated after fit
        self.cluster_labels_ = None
        self.unique_clusters_ = None
        self.kmer_vectors_ = None

    def _fit(self, seqs):
        """Cluster the sequences.

        Args:
            seqs: List of sequences.

        Returns:
            cluster_labels: Cluster label for each sequence.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics.pairwise import cosine_similarity

        np.random.seed(self.random_state)

        if self.verbose:
            print("=" * 70)
            print(f"Similarity-based cluster K-fold split (n_splits={self.n_splits})")
            print("=" * 70)
            print(f"Total samples: {len(seqs)}")
            print(f"Similarity threshold: {self.similarity_threshold}")
            print(f"k-mer length: {self.kmer_size}")

        # Step 1: compute k-mer vectors for all sequences
        if self.verbose:
            print("\n[1/3] Computing k-mer frequency vectors...")

        self.kmer_vectors_ = np.array([compute_kmer_vector(seq, self.kmer_size) for seq in seqs])

        if self.verbose:
            print(f"  k-mer vector dim: {self.kmer_vectors_.shape[1]}")

        # Step 2: pairwise cosine similarity matrix and clustering
        if self.verbose:
            print("\n[2/3] Computing similarity matrix and clustering...")

        similarity_matrix = cosine_similarity(self.kmer_vectors_)

        # Convert to distance matrix (distance = 1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Ensure non-negative (avoid floating-point precision issues)
        distance_matrix = np.maximum(distance_matrix, 0)

        # Ensure exact-zero diagonal (avoid floating-point precision issues)
        np.fill_diagonal(distance_matrix, 0)

        # Hierarchical clustering with average linkage
        Z = linkage(squareform(distance_matrix), method='average')

        # Cut into clusters by distance threshold
        max_distance = 1.0 - self.similarity_threshold
        self.cluster_labels_ = fcluster(Z, t=max_distance, criterion='distance')

        self.unique_clusters_ = np.unique(self.cluster_labels_)
        n_clusters = len(self.unique_clusters_)

        if self.verbose:
            print(f"  Clustering complete: {n_clusters} clusters identified")
            cluster_sizes = [np.sum(self.cluster_labels_ == c) for c in self.unique_clusters_]
            print(f"  Cluster size distribution:")
            print(f"    Smallest cluster: {min(cluster_sizes)} sequences")
            print(f"    Largest cluster: {max(cluster_sizes)} sequences")
            print(f"    Mean cluster size: {np.mean(cluster_sizes):.1f} sequences")

        return self.cluster_labels_

    def split(self, seqs, labels=None):
        """Yield K-fold split indices.

        Uses an improved strategy:
        1. Small clusters (<= n_samples/n_splits) are assigned to a fold as a whole.
        2. Large clusters (> n_samples/n_splits) are further subdivided into
           subclusters.

        Args:
            seqs: List of sequences.
            labels: Corresponding labels (optional; kept for interface compatibility).

        Yields:
            train_indices: Training set indices.
            val_indices: Validation set indices.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Cluster on first call
        if self.cluster_labels_ is None:
            self._fit(seqs)

        n_samples = len(seqs)
        avg_fold_size = n_samples / self.n_splits

        # Build subcluster labels for large clusters
        cluster_sublabels = np.copy(self.cluster_labels_)

        if self.verbose:
            print("\n[3/3] Processing large clusters and splitting...")
            large_clusters = []

        # Identify and subdivide large clusters
        for cluster_id in self.unique_clusters_:
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size > avg_fold_size:
                # Large cluster: needs subdivision
                if self.verbose:
                    large_clusters.append((cluster_id, cluster_size))

                cluster_indices = np.where(cluster_mask)[0]

                # Similarity matrix within this cluster
                cluster_kmers = self.kmer_vectors_[cluster_indices]
                cluster_similarity = cosine_similarity(cluster_kmers)
                cluster_distance = 1 - cluster_similarity
                np.fill_diagonal(cluster_distance, 0)

                # Second-pass clustering with a stricter threshold
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform

                Z_sub = linkage(squareform(cluster_distance), method='average')

                # Use a stricter threshold (0.95) so subcluster members are very similar
                max_distance_sub = 1.0 - min(0.95, self.similarity_threshold + 0.1)
                sub_labels = fcluster(Z_sub, t=max_distance_sub, criterion='distance')

                # Offset sublabels to avoid colliding with original cluster ids
                label_offset = cluster_id * 1000
                for i, sub_label in enumerate(sub_labels):
                    original_idx = cluster_indices[i]
                    cluster_sublabels[original_idx] = label_offset + sub_label

        if self.verbose and large_clusters:
            print(f"  Detected {len(large_clusters)} large clusters (>{avg_fold_size:.0f} sequences)")
            print(f"  Largest cluster size: {max(size for _, size in large_clusters)}")
            print(f"  Subdivided large clusters")

        # Split using subcluster labels
        unique_subclusters = np.unique(cluster_sublabels)
        shuffled_subclusters = np.random.permutation(unique_subclusters)

        subclusters_per_fold = np.array_split(shuffled_subclusters, self.n_splits)

        for fold_idx in range(self.n_splits):
            # This fold's subclusters form the validation set
            val_subclusters = subclusters_per_fold[fold_idx]

            # Remaining subclusters form the training set
            train_subclusters = np.concatenate([
                subclusters_per_fold[i] for i in range(self.n_splits) if i != fold_idx
            ])

            train_indices = np.where([label in train_subclusters for label in cluster_sublabels])[0]
            val_indices = np.where([label in val_subclusters for label in cluster_sublabels])[0]

            if self.verbose:
                # Average similarity between train and validation
                train_kmers = self.kmer_vectors_[train_indices]
                val_kmers = self.kmer_vectors_[val_indices]
                cross_similarity = cosine_similarity(train_kmers, val_kmers)

                print(f"\nFold {fold_idx + 1}/{self.n_splits}:")
                print(f"  Train size: {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)")
                print(f"  Val size: {len(val_indices)} ({len(val_indices)/n_samples*100:.1f}%)")
                print(f"  Train-val mean similarity: {np.mean(cross_similarity):.4f}")
                print(f"  Train-val max similarity: {np.max(cross_similarity):.4f}")

                # Check for highly similar cross-set sequence pairs
                highly_similar_pairs = np.argwhere(cross_similarity > self.similarity_threshold)
                if len(highly_similar_pairs) > 0:
                    print(f"  Warning: found {len(highly_similar_pairs)} highly similar cross-set sequence pairs!")
                else:
                    print(f"  OK: no cross-set sequence pairs with similarity > {self.similarity_threshold}")

            yield train_indices, val_indices

        if self.verbose:
            print("\n" + "=" * 70)
            print("Cluster K-fold split complete!")
            print("=" * 70)


def cluster_based_k_fold_split(seqs, labels, n_splits=5, similarity_threshold=0.8,
                               kmer_size=3, random_state=42, verbose=True):
    """Cluster-based similarity K-fold split (functional interface).

    Convenience function that returns a generator yielding (train_indices,
    val_indices) for each fold.

    Args:
        seqs: List of sequences.
        labels: Corresponding labels.
        n_splits: Number of folds (default 5).
        similarity_threshold: Similarity threshold for hierarchical clustering
            (default 0.8).
        kmer_size: k-mer length (default 3).
        random_state: Random seed (default 42).
        verbose: Whether to print detailed progress (default True).

    Returns:
        Generator yielding (train_indices, val_indices) per fold.

    Example:
        >>> for train_idx, val_idx in cluster_based_k_fold_split(seqs, labels, n_splits=5):
        ...     train_seqs = [seqs[i] for i in train_idx]
        ...     val_seqs = [seqs[i] for i in val_idx]
        ...     # train and validate...
    """
    kfold = ClusterBasedKFold(
        n_splits=n_splits,
        similarity_threshold=similarity_threshold,
        kmer_size=kmer_size,
        random_state=random_state,
        verbose=verbose
    )

    return kfold.split(seqs, labels)
