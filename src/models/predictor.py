import random
import os
import sys
import functools
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import torch
from torch import nn
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)

from sklearn.preprocessing import LabelEncoder

# ========== Model architecture definitions ==========
# Register new architectures here using the @ModelFactory.register decorator.

class ModelFactory:
    """Factory for registering and retrieving model architectures."""
    _models = defaultdict(dict)

    @classmethod
    def register(cls, name):
        """Decorator that registers a model architecture."""
        def wrapper(model_class):
            cls._models[name] = model_class
            print(f"Registered model: {name}")
            return model_class
        return wrapper

    @classmethod
    def create(cls, name, **kwargs):
        """Create a model instance of the registered name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model type: {name}")
        return cls._models[name](**kwargs)
    def get_model(cls, name, **kwargs):
        """Return a registered model."""
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered")
        return cls._models[name](**kwargs)

from src.utils.module import BidirectionalLSTM


# ========== Ablation support ==========
# Legal ablation variants (CNN–Attention–BiLSTM)
ABLATION_VARIANTS = ('full', 'no_attention', 'no_bilstm', 'no_attention_no_bilstm')


class PositionWiseLinear(nn.Module):
    """Position-wise linear mapping used as the BiLSTM ablation replacement.

    Input/output tensor conventions match ``BidirectionalLSTM`` exactly:
      input ``[B, C_in, T]``  ->  output ``[B, C_out, T]``
    Applies only a ``C_in -> C_out`` linear map at each position with no sequence
    mixing, so it can be independently initialized and retrained (rather than being
    zeroed out at inference time).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: [B, C_in, T]
        x = x.permute(0, 2, 1)            # [B, T, C_in]
        x = self.linear(x)                # [B, T, C_out]
        x = x.permute(0, 2, 1)            # [B, C_out, T]
        return x


@ModelFactory.register('conv')
class ConvPredictor(nn.Module):
    """Bidirectional LSTM with an attention mechanism.

    ``ablation_variant`` controls the ablated architecture (default ``full`` is the
    full architecture):

    - ``full``                    : Conv -> Transformer -> BiLSTM -> MaxPool
    - ``no_attention``            : replace the TransformerEncoderLayer with ``nn.Identity``
    - ``no_bilstm``               : replace the BiLSTM with position-wise ``PositionWiseLinear`` (no sequence mixing)
    - ``no_attention_no_bilstm``  : apply both of the replacements above
    """
    def __init__(self, seq_len=None, motif_conv_hidden=256, conv_hidden=128, n_heads=16, conv_width_motif=5, dropout_rate=0.2,
                 num_classes=None, vocab_size=None, task_type='regression', ablation_variant='full'):
        super().__init__()
        if ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown ablation_variant: {ablation_variant!r}. "
                f"Expected one of {ABLATION_VARIANTS}."
            )
        self.ablation_variant = ablation_variant
        self.vocab_size = vocab_size
        self.task_type = task_type
        self.num_classes = num_classes if task_type == 'classification' else 1

        if vocab_size is not None:
            # Embedding layer for tokenized input
            self.embedding = nn.Embedding(vocab_size, conv_hidden)
            self.conv1 = None
            self.norm1 = None
            self.relu1 = None
            self.conv2 = None
            self.norm2 = None
            self.relu2 = None
        else:
            # Convolutional layers for one-hot input
            self.conv1 = nn.Conv1d(in_channels=4, out_channels=motif_conv_hidden, kernel_size=conv_width_motif, padding='same')
            self.norm1 = nn.BatchNorm1d(motif_conv_hidden)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=motif_conv_hidden, out_channels=conv_hidden, kernel_size=conv_width_motif, padding='same')
            self.norm2 = nn.BatchNorm1d(conv_hidden)
            self.relu2 = nn.ReLU()
            self.embedding = None

        # ---- Ablation branch: attention ----
        use_attention = ablation_variant in ('full', 'no_bilstm')
        if use_attention:
            self.attention = nn.TransformerEncoderLayer(d_model=conv_hidden, nhead=n_heads, batch_first=True)
        else:
            # no_attention / no_attention_no_bilstm: Identity placeholder preserving tensor dimensions
            self.attention = nn.Identity()

        # ---- Ablation branch: bilstm ----
        use_bilstm = ablation_variant in ('full', 'no_attention')
        if use_bilstm:
            self.bilstm = BidirectionalLSTM(input_size=conv_hidden, hidden_size=conv_hidden, output_size=int(conv_hidden // 4))
        else:
            # no_bilstm / no_attention_no_bilstm: position-wise linear map, output channels still conv_hidden//4
            self.bilstm = PositionWiseLinear(conv_hidden, int(conv_hidden // 4))
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # global max pooling
        self.dense1 = nn.Linear(int(conv_hidden // 4), conv_hidden)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(conv_hidden, conv_hidden)
        self.relu4 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        # Output head depends on task type
        if task_type == 'classification':
            self.dense3 = nn.Linear(conv_hidden, 1)  # scalar output fed to ORNet
            from src.utils.module import ORNet
            self.ornet = ORNet(1, num_classes)
        else:
            self.dense3 = nn.Linear(conv_hidden, 1)  # regression output
            self.ornet = None

    def forward(self, x):
        if self.vocab_size is not None:
            # x: [batch_size, seq_len] (tokenized)
            x = self.embedding(x)  # [batch_size, seq_len, conv_hidden]
        else:
            # x: [batch_size, 4, seq_len] (one-hot)
            x = self.relu1(self.norm1(self.conv1(x)))
            x = self.relu2(self.norm2(self.conv2(x)))
            x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]

        x = self.attention(x)  # [batch_size, seq_len, conv_hidden]

        x = x.permute(0, 2, 1)  # [batch_size, conv_hidden, seq_len]
        x = self.bilstm(x)  # [batch_size, channels, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, channels]
        x = self.relu3(self.dense1(x))
        x = self.relu4(self.dense2(self.drop1(x)))
        x = self.dense3(self.drop2(x))
        if self.task_type == 'regression':
            return x.squeeze(-1)  # [batch_size] - regression output
        else:
            # Classification: convert to ordinal-regression logits via ORNet
            logits = self.ornet(x)
            return logits  # [batch_size, num_classes-1]

@ModelFactory.register('1dcnn')
class OneDCNN(nn.Module):
    """1D-CNN for short DNA sequence regression/classification.

    Lightweight three-layer 1D-CNN with global average pooling,
    optimized for short regulatory elements (15-50 bp).
    """
    def __init__(self, seq_len=15, dropout_rate=0.2, vocab_size=None, num_classes=None, task_type='regression', **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.task_type = task_type
        self.num_classes = num_classes if task_type == 'classification' else 1

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, 64)
            in_channels = 64
        else:
            self.embedding = None
            in_channels = 4

        # Key change: drop pooling layers and use stride=1 convolutions
        self.conv1 = nn.Conv1d(in_channels, 96, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(96)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(192)

        # Use global pooling instead of multiple MaxPool layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Simplified fully-connected head; output depends on task type
        if task_type == 'classification':
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)  # scalar output fed to ORNet
            )
            from src.utils.module import ORNet
            self.ornet = ORNet(1, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
            self.ornet = None

    def forward(self, x):
        if self.vocab_size is not None:
            x = self.embedding(x).permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        if self.task_type == 'regression':
            return x.squeeze(-1)
        else:
            # Classification: convert to ordinal-regression logits via ORNet
            logits = self.ornet(x)
            return logits  # [batch_size, num_classes-1]
