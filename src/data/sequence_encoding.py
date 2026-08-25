import numpy as np

def rna_clean_only(string):
    '''
    Clean up RNA fasta and SILVA "fasta" formatting. Leave '=' as only
    remaining non-RNA character that could be used as padding later.
    Replace the rare "Purine" R with A and "Pyrimidine" Y with U.
    Replace K,B,H with U; M,S with C; Z,N,W,V,D,n with A.
    Remove newline characters.
    Also replace spaces from SILVA, but keep dashes for MSA.
    '''
    for char in ['R', 'N', 'W', 'V', 'D', 'Z', 'n', 'a']:
        string = string.replace(char, 'A')
    for char in ['Y', 'K', 'B', 'H', 'u', 'T']:
        string = string.replace(char, 'U')
    for char in ['M', 'S', 'c']:
        string = string.replace(char, 'C')
    string = string.replace('g', 'G')
    for char in ['\n', ' ', '.', ]: # Keep '-' in string for MSA.
        string = string.replace(char, '')

    return(string)

def single_encoding(sequence):
    '''
    A 7-token library for RNA nucleotides that includes '-' for padding.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach.
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    chars = ['A', 'U', 'G', 'C']
    chars += ['5', '3', '-']

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    train_stoi = []
    for i in range(len(sequence)):
        w = stoi.get(sequence[i], stoi['-'])  # Use '-' for unknown chars
        train_stoi.append(w)

    return stoi, itos, train_stoi

def pairs_encoding(sequence):
    '''
    A 25-token library for RNA nucleotide pairs that includes '--' for padding.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach.
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    nucleotides = ['A', 'U', 'G', 'C']

    # Generate all possible pairs from nucleotides list
    chars = [a+b for a in nucleotides for b in nucleotides]

    # Now add the combinations with '5' in the first position and '3' in the second.
    chars.extend(['5' + b for b in nucleotides])
    chars.extend([a + '3' for a in nucleotides])
    chars += ['--']

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    train_stoi = []
    for i in range(len(sequence) - 1):
        pair = sequence[i:i+2]
        if not '--' in pair and not '-5' in pair and not '3-' in pair:
            w = stoi.get(pair, stoi['--'])
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('--')
            train_stoi.append(w)

    return stoi, itos, train_stoi

def triples_encoding(sequence):
    '''
    A token library for RNA nucleotide triples that includes '---' for padding and excludes double-dash ends like '--A' and 'A--'.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach.
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    nucleotides = ['A', 'U', 'G', 'C']

    # Generate all possible triples from nucleotides list
    chars = [a+b+c for a in nucleotides for b in nucleotides for c in nucleotides]

    # Add combinations with '5' in the first position and '3' in the third position.
    chars.extend(['5' + b + c for b in nucleotides for c in nucleotides])
    chars.extend([a + b + '3' for a in nucleotides for b in nucleotides])

    chars += ['---']

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    train_stoi = []
    for i in range(len(sequence) - 2):
        triple = sequence[i:i+3]
        if not '--' in triple and not '-5' in triple and not '3-' in triple:
            w = stoi.get(triple, stoi['---'])
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('---')
            train_stoi.append(w)

    return stoi, itos, train_stoi

# Encoding type mapping
ENCODING_FUNCTIONS = {
    'single': single_encoding,
    'pairs': pairs_encoding,
    'triples': triples_encoding
}

def encode_sequences(sequences, encoding_type='single'):
    '''
    Encode a list of sequences using the specified encoding type.

    Args:
        sequences: List of RNA sequences (strings)
        encoding_type: 'single', 'pairs', or 'triples'

    Returns:
        encoded_sequences: List of encoded sequences (numpy arrays)
        stoi: String to int mapping
        itos: Int to string mapping
        vocab_size: Size of vocabulary
    '''
    if encoding_type not in ENCODING_FUNCTIONS:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    # Clean all sequences first
    cleaned_sequences = [rna_clean_only(seq) for seq in sequences]

    # Use the first sequence to get vocab info
    stoi, itos, _ = ENCODING_FUNCTIONS[encoding_type](cleaned_sequences[0])
    vocab_size = len(itos)

    encoded_sequences = []
    for seq in cleaned_sequences:
        _, _, encoded = ENCODING_FUNCTIONS[encoding_type](seq)
        encoded_sequences.append(np.array(encoded, dtype=np.int32))

    return encoded_sequences, stoi, itos, vocab_size