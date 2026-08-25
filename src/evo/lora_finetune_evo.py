#!/usr/bin/env python3
"""
Fine-tune Evo model for promoter strength prediction using LoRA.

This script adds a regression head to Evo model and fine-tunes it
with LoRA (Low-Rank Adaptation) to predict promoter strength from DNA sequences.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
# Import cluster-based split utilities from DeepBioParts
from src.utils.data import cluster_based_split, compute_kmer_vector, ClusterBasedKFold
# stripedhyena / transformers are used only inside load_evo_from_local; they are imported
# lazily so this module remains importable in environments lacking the Evo dependencies
# (needed by unit tests that construct EvoWithRegressionHead).
import yaml
import random


def dna_sequence_to_tokens(sequence, max_len=40):
    """Encode a DNA sequence as a list of token IDs for Evo model inference.

    Args:
        sequence: DNA sequence string
        max_len: Maximum sequence length (truncation or padding)

    Returns:
        list[int]: Token ID list of length max_len
    """
    from stripedhyena.tokenizer import CharLevelTokenizer
    tokenizer = CharLevelTokenizer(512)
    pad_id = getattr(tokenizer, 'pad_id',
                     getattr(tokenizer, 'pad_token_id', 0))
    tokens = tokenizer.tokenize(sequence)
    tokens = tokens[:max_len]
    if len(tokens) < max_len:
        tokens = tokens + [pad_id] * (max_len - len(tokens))
    return tokens


def reverse_complement(dna_sequence):
    """Generate reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(dna_sequence))


def random_kmer_mask(dna_sequence, k=3, mask_ratio=0.1, mask_token='N'):
    """Randomly mask k-mers in DNA sequence for data augmentation."""
    seq_list = list(dna_sequence)
    num_kmers_to_mask = max(1, int(len(dna_sequence) * mask_ratio / k))
    for _ in range(num_kmers_to_mask):
        start = random.randint(0, len(dna_sequence) - k)
        for i in range(start, min(start + k, len(dna_sequence))):
            seq_list[i] = mask_token
    return ''.join(seq_list)


class PromoterDataset(Dataset):
    """
    Dataset for promoter sequences with strength labels or binary classification labels.

    FIX #6: Added safe fallback for pad_id attribute (supports both pad_id and pad_token_id)
    FIX #1: Added attention_mask generation to prevent padding leakage in attention pooling
    """

    def __init__(self, sequences, labels, tokenizer, max_len=40, task_type='regression',
                 augment=False, augment_prob=0.3):
        """
        Args:
            sequences: List of DNA sequences
            labels: Array of labels (activity values for regression, class indices for classification)
            tokenizer: Tokenizer for sequences
            max_len: Maximum sequence length
            task_type: 'regression' or 'classification'
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation per sample
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_type = task_type
        self.augment = augment
        self.augment_prob = augment_prob

        # FIX #6: Safe fallback for pad_id - supports both HuggingFace and custom tokenizers
        self.pad_id = getattr(self.tokenizer, 'pad_id',
                             getattr(self.tokenizer, 'pad_token_id', 0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Apply data augmentation during training
        if self.augment and random.random() < self.augment_prob:
            # Randomly choose augmentation strategy
            aug_type = random.choice(['reverse_complement', 'kmer_mask'])
            if aug_type == 'reverse_complement':
                seq = reverse_complement(seq)
            elif aug_type == 'kmer_mask':
                seq = random_kmer_mask(seq, k=3, mask_ratio=0.1)

        # Tokenize sequence
        tokens = self.tokenizer.tokenize(seq)
        tokens = tokens[:self.max_len]  # Truncate to max_len

        # FIX #1: Generate attention mask (1 for real tokens, 0 for padding)
        seq_len = len(tokens)
        attention_mask = torch.ones(self.max_len, dtype=torch.long)
        if seq_len < self.max_len:
            attention_mask[seq_len:] = 0  # Mark padding positions

        # Pad sequence
        if len(tokens) < self.max_len:
            tokens = tokens + [self.pad_id] * (self.max_len - len(tokens))

        # Process label based on task type
        if self.task_type == 'classification':
            # For classification, label should be a class index (0 or 1)
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            # For regression, use original label directly
            label_tensor = torch.tensor(label, dtype=torch.float32)

        # FIX #1: Return attention_mask along with tokens and label
        return torch.tensor(tokens, dtype=torch.long), label_tensor, attention_mask


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) for Linear layers.
    y = Wx + BAx, where B and A are low-rank matrices.

    Memory-efficient version: stores reference to original weight instead of copying.
    """

    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dtype = dtype

        # Reference to original weight (NOT a copy!)
        self.original_weight = None
        self.original_bias = None

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        self.lora_dropout = nn.Dropout(dropout)

        # Reset LoRA B to zero
        nn.init.zeros_(self.lora_B)

    def set_original_weight(self, weight, bias=None):
        """Set reference to original weight (does not copy data)."""
        self.original_weight = weight
        self.original_bias = bias

    def forward(self, x):
        # Original linear transformation (using reference, not copy)
        result = nn.functional.linear(x, self.original_weight, self.original_bias)

        # LoRA adaptation: x -> A -> B -> output
        # A: [rank, in_features], B: [out_features, rank]
        # x: [batch, in_features]
        # First: x @ A.T -> [batch, rank]
        # Second: result @ B.T -> [batch, out_features]
        lora_result = nn.functional.linear(
            self.lora_dropout(x),
            self.lora_A
        )  # [batch, rank] (x @ A.T)
        lora_result = nn.functional.linear(lora_result, self.lora_B)  # [batch, out_features] (x @ A.T @ B.T)

        return result + self.scaling * lora_result
    
    def get_lora_parameters(self):
        """Return only LoRA parameters for training."""
        return [self.lora_A, self.lora_B]


def apply_lora_to_linear(linear_module, rank=16, alpha=32, dropout=0.1):
    """
    Replace a Linear module with a LoRA-enabled version.

    LoRA parameters use the same dtype as the original module (typically bfloat16 for Evo).
    GradScaler (PyTorch 1.12+) supports bfloat16 gradients.
    """
    # Use the same dtype as the original module for LoRA parameters
    # This ensures gradient flow and compatibility with the base model
    dtype = linear_module.weight.dtype
    lora_linear = LoRALinear(
        linear_module.in_features,
        linear_module.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        dtype=dtype
    )
    lora_linear.set_original_weight(linear_module.weight, linear_module.bias)
    return lora_linear


class AttentionPoolingRegression(nn.Module):
    """
    FIX #1: Attention Pooling + Simple Projection Regression Head.

    Now accepts and uses attention_mask to prevent padding tokens from
    contaminating the pooled representation. Padding tokens are masked out
    before softmax so their attention weights become exactly 0.

    Uses learnable attention weights to pool sequence hidden states,
    followed by a simple two-layer MLP for regression.

    ``pooling_mode``:
    - ``attention`` (default): masked attention pooling
    - ``mean``: masked mean pooling (via attention_mask), followed by the same
      regression MLP as the full model
    """

    def __init__(self, hidden_dim=4096, intermediate_dim=512, dropout=0.1, pooling_mode='attention'):
        super().__init__()
        if pooling_mode not in ('attention', 'mean'):
            raise ValueError(f"Unknown pooling_mode: {pooling_mode!r}. Expected 'attention' or 'mean'.")
        self.pooling_mode = pooling_mode

        # Attention weight computation (used in attention mode only)
        if pooling_mode == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
        else:
            self.attention = None

        # Simple regression projection (two-layer MLP) — shared by both pooling modes
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1)
        )

    def _mean_pool(self, hidden_states, attention_mask):
        """Masked mean pooling over the sequence dimension."""
        if attention_mask is None:
            # Degenerate case: without a mask, falls back to a plain mean
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        return masked.sum(dim=1) / denom  # [B, hidden_dim]

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding (FIX #1)

        Returns:
            output: [batch] predicted strength values
        """
        if self.pooling_mode == 'mean':
            pooled = self._mean_pool(hidden_states, attention_mask)
        else:
            # Compute attention weights
            attn_weights = self.attention(hidden_states)  # [batch, seq_len, 1]

            # FIX #1: Apply attention mask before softmax to prevent padding leakage
            if attention_mask is not None:
                # Expand mask to match attn_weights shape: [batch, seq_len, 1]
                mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                # Set padding positions to -inf so they become 0 after softmax
                attn_weights = attn_weights.masked_fill(mask_expanded == 0, -1e4)

            attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over seq_len

            # Weighted pooling
            pooled = (hidden_states * attn_weights).sum(dim=1)  # [batch, hidden_dim]

        # Regression
        output = self.regression(pooled)  # [batch, 1]
        return output.squeeze(-1)  # [batch]


class AttentionPoolingClassification(nn.Module):
    """
    FIX #1: Attention Pooling + Binary Classification Head.

    Now accepts and uses attention_mask to prevent padding tokens from
    contaminating the pooled representation. Padding tokens are masked out
    before softmax so their attention weights become exactly 0.

    Uses learnable attention weights to pool sequence hidden states,
    followed by a simple two-layer MLP for binary classification.

    ``pooling_mode``:
    - ``attention`` (default): masked attention pooling
    - ``mean``: masked mean pooling, followed by the same classification MLP
      as the full model
    """

    def __init__(self, hidden_dim=4096, intermediate_dim=512, dropout=0.1, pooling_mode='attention'):
        super().__init__()
        if pooling_mode not in ('attention', 'mean'):
            raise ValueError(f"Unknown pooling_mode: {pooling_mode!r}. Expected 'attention' or 'mean'.")
        self.pooling_mode = pooling_mode

        # Attention weight computation (used in attention mode only)
        if pooling_mode == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
        else:
            self.attention = None

        # Simple classification projection (two-layer MLP) — shared by both pooling modes
        self.classification = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 2)  # Binary classification: 2 output classes
        )

    def _mean_pool(self, hidden_states, attention_mask):
        """Masked mean pooling over the sequence dimension."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        return masked.sum(dim=1) / denom  # [B, hidden_dim]

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding (FIX #1)

        Returns:
            logits: [batch, 2] classification logits for 2 classes
        """
        if self.pooling_mode == 'mean':
            pooled = self._mean_pool(hidden_states, attention_mask)
        else:
            # Compute attention weights
            attn_weights = self.attention(hidden_states)  # [batch, seq_len, 1]

            # FIX #1: Apply attention mask before softmax to prevent padding leakage
            if attention_mask is not None:
                # Expand mask to match attn_weights shape: [batch, seq_len, 1]
                mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                # Set padding positions to -inf so they become 0 after softmax
                attn_weights = attn_weights.masked_fill(mask_expanded == 0, -1e4)

            attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over seq_len

            # Weighted pooling
            pooled = (hidden_states * attn_weights).sum(dim=1)  # [batch, hidden_dim]

        # Classification
        logits = self.classification(pooled)  # [batch, 2]
        return logits


class EvoWithRegressionHead(nn.Module):
    """
    Evo model with regression head for promoter strength prediction
    or binary classification head for terminator prediction.
    """

    def __init__(self, evo_model, hidden_dim=512, dropout=0.2, use_lora=True,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1, task_type='regression',
                 use_gradient_checkpointing=False, evo_adaptation=None, pooling_mode='attention'):
        """
        Args:
            task_type: 'regression' for promoter/rbs strength prediction,
                      'classification' for terminator binary classification
            evo_adaptation: How the Evo backbone is adapted (ablation). When None, it is
                inferred from ``use_lora`` for backward compatibility
                (use_lora=True -> 'lora', use_lora=False -> 'partial_ft').
                - 'lora'       : LoRA injection
                - 'head_only'  : freeze the entire Evo backbone, no LoRA; train pooling + output head only
                - 'partial_ft' : unfreeze the last 4 blocks
            pooling_mode: Pooling strategy (ablation).
                - 'attention' : masked attention pooling (default)
                - 'mean'      : masked mean pooling, followed by the same regression/classification MLP as the full model
        """
        super().__init__()
        # Resolve evo_adaptation: if unspecified, infer from use_lora for backward compatibility
        if evo_adaptation is None:
            evo_adaptation = 'lora' if use_lora else 'partial_ft'
        if evo_adaptation not in ('lora', 'head_only', 'partial_ft'):
            raise ValueError(
                f"Unknown evo_adaptation: {evo_adaptation!r}. Expected one of ('lora', 'head_only', 'partial_ft')."
            )
        if pooling_mode not in ('attention', 'mean'):
            raise ValueError(f"Unknown pooling_mode: {pooling_mode!r}. Expected 'attention' or 'mean'.")
        self.evo_adaptation = evo_adaptation
        self.pooling_mode = pooling_mode
        self.task_type = task_type
        self.evo = evo_model.model
        self.tokenizer = evo_model.tokenizer
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Always freeze all Evo backbone parameters first
        for param in self.evo.parameters():
            param.requires_grad = False

        if evo_adaptation == 'lora':
            # Apply LoRA to the Evo backbone
            print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

            # Apply LoRA to linear layers in StripedHyena blocks
            lora_count = 0
            for i, block in enumerate(self.evo.blocks):
                # Try to find and replace linear layers in attention
                if hasattr(block, 'attention'):
                    for name, module in list(block.attention.named_children()):
                        if isinstance(module, nn.Linear):
                            print(f"  Applying LoRA to blocks.{i}.attention.{name}")
                            lora_module = apply_lora_to_linear(module, lora_r, lora_alpha, lora_dropout)
                            setattr(block.attention, name, lora_module)
                            lora_count += 1

                # Try to find and replace linear layers in FFN
                if hasattr(block, 'ffn'):
                    for name, module in list(block.ffn.named_children()):
                        if isinstance(module, nn.Linear):
                            print(f"  Applying LoRA to blocks.{i}.ffn.{name}")
                            lora_module = apply_lora_to_linear(module, lora_r, lora_alpha, lora_dropout)
                            setattr(block.ffn, name, lora_module)
                            lora_count += 1

                # Also check for linear layers directly in block
                for name, module in list(block.named_children()):
                    if isinstance(module, nn.Linear) and name not in ['attention', 'ffn']:
                        print(f"  Applying LoRA to blocks.{i}.{name}")
                        lora_module = apply_lora_to_linear(module, lora_r, lora_alpha, lora_dropout)
                        setattr(block, name, lora_module)
                        lora_count += 1

            print(f"Applied LoRA to {lora_count} linear layers")

            # Enable gradient checkpointing if requested
            if self.use_gradient_checkpointing:
                print("Enabling gradient checkpointing...")
                # StripedHyena blocks support gradient checkpointing
                if hasattr(self.evo, 'blocks'):
                    for block in self.evo.blocks:
                        block.checkpoint = True  # Enable checkpointing for each block
                print("Gradient checkpointing enabled (trades computation for memory)")

            # Count trainable parameters
            lora_params = 0
            total_params = 0
            for param in self.evo.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    lora_params += param.numel()

            print(f"LoRA trainable parameters: {lora_params:,} ({100 * lora_params / total_params:.2f}%)")
            print(f"Total parameters: {total_params:,}")
        elif evo_adaptation == 'partial_ft':
            # Unfreeze the last 4 blocks (partial fine-tuning, no LoRA)
            print("Using partial freezing strategy (not LoRA)")
            if hasattr(self.evo, 'blocks'):
                for block in self.evo.blocks[-4:]:
                    for param in block.parameters():
                        param.requires_grad = True
        else:  # head_only: freeze the whole backbone, no LoRA, train the head only
            print("Using head-only adaptation: freezing all Evo backbone params, training head only")

        # Create appropriate head based on task type
        # Evo-1-8k-base has hidden_size of 4096
        if task_type == 'classification':
            self.head = AttentionPoolingClassification(
                hidden_dim=4096,
                intermediate_dim=hidden_dim,
                dropout=dropout,
                pooling_mode=pooling_mode
            )
            print(f"Using binary classification head (pooling_mode={pooling_mode})")
        else:  # regression
            self.head = AttentionPoolingRegression(
                hidden_dim=4096,
                intermediate_dim=hidden_dim,
                dropout=dropout,
                pooling_mode=pooling_mode
            )
            print(f"Using regression head (pooling_mode={pooling_mode})")

        # Print head parameter count
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Head parameters: {head_params:,}")

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through Evo model with task-specific head.

        FIX #5: Added `x = x.requires_grad_()` before checkpointed block loop to ensure
        gradient graph connectivity when using gradient checkpointing with frozen base layers.
        FIX #1: Now accepts and passes attention_mask to the head to prevent padding leakage.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding (FIX #1)

        Returns:
            output: [batch] for regression, [batch, 2] for classification
        """
        # Get Evo output - we need to extract hidden states, not logits
        # StripedHyena returns (logits, cache) where logits is [batch, seq_len, vocab_size]
        # We need to access the hidden states from the model

        # Enable gradient checkpointing if configured (for training)
        if self.use_gradient_checkpointing and self.training:
            # Temporarily enable gradient checkpointing for Evo blocks
            for block in self.evo.blocks:
                if hasattr(block, 'checkpoint'):
                    block.checkpoint = True

        # Use gradient checkpointing wrapper to reduce memory
        if self.use_gradient_checkpointing and self.training:
            # Custom checkpointed forward pass
            def run_block(block_module, hidden_state):
                hidden_state, _ = block_module(hidden_state)
                return hidden_state

            # Get embeddings first
            x = self.evo.embedding_layer(input_ids)  # [batch, seq_len, d_model]

            # FIX #5: Explicitly enable gradient tracking for checkpointed computation
            # When the base embedding layer is frozen, x has requires_grad=False.
            # We need to re-enable it so LoRA parameters inside blocks can build the backward graph.
            x = x.requires_grad_()

            # Pass through blocks with checkpointing
            for block in self.evo.blocks:
                x = torch.utils.checkpoint.checkpoint(run_block, block, x,
                                                       use_reentrant=False)

            # Apply final norm
            x = self.evo.norm(x)  # [batch, seq_len, d_model]
        else:
            # Get embeddings first
            x = self.evo.embedding_layer(input_ids)  # [batch, seq_len, d_model]

            # Pass through blocks
            for block in self.evo.blocks:
                x, _ = block(x)

            # Apply final norm
            x = self.evo.norm(x)  # [batch, seq_len, d_model]

        # Cast to float32 for head (BFloat16 not supported by all ops)
        x = x.float()

        # FIX #1: Pass attention_mask to head to prevent padding leakage
        output = self.head(x, attention_mask)  # [batch] for regression, [batch, 2] for classification

        return output


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, task_type='regression',
                gradient_accumulation_steps=1):
    """
    Train for one epoch with gradient accumulation support.

    FIX #1: Now handles attention_mask from dataset to prevent padding leakage.
    FIX #3: Added final optimizer step after loop to handle remaining gradients
            when batches aren't perfectly divisible by gradient_accumulation_steps.
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    accumulation_counter = 0

    for batch_data in tqdm(dataloader, desc="Training", leave=False):
        # FIX #1: Unpack attention_mask along with input_ids and labels
        if len(batch_data) == 3:
            input_ids, labels, attention_mask = batch_data
        else:
            # Backward compatibility: handle old dataset format without attention_mask
            input_ids, labels = batch_data
            attention_mask = None

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                # FIX #1: Pass attention_mask to model
                predictions = model(input_ids, attention_mask)

                if task_type == 'classification':
                    # For classification, labels are class indices, predictions are [batch, 2]
                    loss = criterion(predictions, labels)
                else:
                    # For regression, labels are continuous values, predictions are [batch]
                    loss = criterion(predictions, labels)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            accumulation_counter += 1
            # Only step optimizer after accumulating gradients
            if accumulation_counter % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # For BFloat16 training: use autocast but NO GradScaler
            # BFloat16 has the same exponent range as FP32, so gradient scaling is not needed
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # FIX #1: Pass attention_mask to model
                predictions = model(input_ids, attention_mask)

                if task_type == 'classification':
                    loss = criterion(predictions, labels)
                else:
                    loss = criterion(predictions, labels)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            loss.backward()

            accumulation_counter += 1
            # Only step optimizer after accumulating gradients
            if accumulation_counter % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Clear cache to reduce memory fragmentation
                torch.cuda.empty_cache()

        total_loss += loss.item()

        # Delete intermediate tensors to free memory
        del predictions, loss
        if 'input_ids' in locals():
            del input_ids
        if 'labels' in locals():
            del labels
        if 'attention_mask' in locals() and attention_mask is not None:
            del attention_mask

    # FIX #3: Handle remaining gradients if batches aren't perfectly divisible
    # If accumulation_counter > 0 after loop, we have unstepped gradients
    if accumulation_counter % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, task_type='regression'):
    """
    Validate the model.

    FIX #1: Now handles attention_mask from dataset to prevent padding leakage.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation", leave=False):
            # FIX #1: Unpack attention_mask along with input_ids and labels
            if len(batch_data) == 3:
                input_ids, labels, attention_mask = batch_data
            else:
                # Backward compatibility: handle old dataset format without attention_mask
                input_ids, labels = batch_data
                attention_mask = None

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # FIX #1: Pass attention_mask to model
            predictions = model(input_ids, attention_mask)

            if task_type == 'classification':
                loss = criterion(predictions, labels)
                # Get predicted class from logits
                pred_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(pred_classes.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
            else:
                loss = criterion(predictions, labels)
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            total_loss += loss.item()

            # Clean up tensors to free memory
            del predictions, loss, input_ids, labels
            if attention_mask is not None:
                del attention_mask

    # Clear cache after validation
    torch.cuda.empty_cache()

    # Calculate metrics based on task type
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        return total_loss / len(dataloader), accuracy, precision, recall, f1
    else:
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        # Calculate correlation coefficients
        pearson_corr, pearson_p = pearsonr(all_targets, all_preds)
        spearman_corr, spearman_p = spearmanr(all_targets, all_preds)

        return total_loss / len(dataloader), mse, mae, r2, pearson_corr, spearman_corr


def load_evo_from_local(local_model_path, model_name='evo-1.5-8k-base', device='cuda:0'):
    """
    Load Evo model from local directory.

    Args:
        local_model_path: Path to local model directory (if None, will try to find in ./models/)
        model_name: Model name for config selection
        device: Device to load model on

    Returns:
        evo_model: Evo model instance
        tokenizer: Character level tokenizer
    """
    # Handle None local_model_path by trying default location
    if local_model_path is None:
        default_local_path = os.path.join(os.path.dirname(__file__), 'models', model_name)
        if os.path.exists(default_local_path):
            local_model_path = default_local_path
            print(f"Found local model at: {local_model_path}")
        else:
            raise FileNotFoundError(
                f"Local model path not provided and default path not found: {default_local_path}\n"
                f"Please provide --local-model-path or ensure model exists at {default_local_path}"
            )

    print(f"Loading Evo model from local path: {local_model_path}")

    # Lazy import: needed only when actually loading Evo weights
    from stripedhyena.model import StripedHyena
    from stripedhyena.utils import dotdict

    # Assign config path.
    if model_name == 'evo-1-8k-base' or \
       model_name == 'evo-1-8k-crispr' or \
       model_name == 'evo-1-8k-transposon' or \
       model_name == 'evo-1.5-8k-base':
        config_filename = 'evo-1-8k-base_inference.yml'
    elif model_name == 'evo-1-131k-base':
        config_filename = 'evo-1-131k-base_inference.yml'
    else:
        raise ValueError(f'Invalid model name {model_name}')
    
    # Load SH config - try multiple paths
    config_paths = [
        # Path 1: evo package configs directory (in DeepBioParts/evo/evo/configs)
        os.path.join(os.path.dirname(__file__), 'evo', 'configs', config_filename),
        # Path 2: relative to script (fallback)
        os.path.join(os.path.dirname(__file__), 'configs', config_filename),
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            print(f"Using config file: {config_path}")
            break
    
    if config_path is None:
        raise FileNotFoundError(
            f"Config file '{config_filename}' not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in config_paths)
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    global_config = dotdict(config, Loader=yaml.FullLoader)
    
    # Load SH Model.
    sh_model = StripedHyena(global_config)
    
    # Load model weights from local directory
    print(f"Loading model weights from: {local_model_path}")
    
    # Try to load from safetensors first
    safetensors_file = os.path.join(local_model_path, 'model.safetensors')
    pytorch_file = os.path.join(local_model_path, 'pytorch_model.bin')
    index_file = os.path.join(local_model_path, 'pytorch_model.bin.index.json')
    
    if os.path.exists(safetensors_file):
        print(f"Loading from safetensors: {safetensors_file}")
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(safetensors_file)
    elif os.path.exists(index_file):
        # Load sharded model
        print(f"Loading sharded pytorch model with index: {index_file}")
        import json
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        # Load all shards and merge
        state_dict = {}
        for key, filename in index['weight_map'].items():
            shard_path = os.path.join(local_model_path, filename)
            if shard_path not in state_dict:
                print(f"Loading shard: {filename}")
                shard_state_dict = torch.load(shard_path, map_location='cpu')
                state_dict[shard_path] = shard_state_dict
        
        # Merge shards into a single state_dict
        merged_state_dict = {}
        for key, filename in index['weight_map'].items():
            shard_path = os.path.join(local_model_path, filename)
            merged_state_dict[key] = state_dict[shard_path][key]
        
        state_dict = merged_state_dict
    elif os.path.exists(pytorch_file):
        print(f"Loading from pytorch bin: {pytorch_file}")
        state_dict = torch.load(pytorch_file, map_location='cpu')
    else:
        raise FileNotFoundError(
            f"Model weights not found in {local_model_path}. "
            f"Expected 'model.safetensors', 'pytorch_model.bin', or 'pytorch_model.bin.index.json'."
        )
    
    # The state_dict from HuggingFace has a different structure
    # We need to remove 'backbone.' prefix from keys if present
    # Check if all keys have 'backbone.' prefix
    if all(k.startswith('backbone.') for k in state_dict.keys()):
        # Remove 'backbone.' prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('backbone.', '', 1)
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    elif 'backbone' in state_dict:
        # Handle nested structure where 'backbone' is a top-level key
        state_dict = state_dict['backbone']
    elif '"backbone"' in state_dict:
        # Handle nested structure where '"backbone"' is a top-level key
        state_dict = state_dict['"backbone"']
    
    sh_model.load_state_dict(state_dict, strict=True)
    sh_model.to_bfloat16_except_poles_residues()
    if device is not None:
        sh_model = sh_model.to(device)
    
    # Create Evo-like object
    class LocalEvo:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
    
    from stripedhyena.tokenizer import CharLevelTokenizer
    tokenizer = CharLevelTokenizer(512)
    
    return LocalEvo(sh_model, tokenizer)


def finetune_evo_promoter_strength(
    data_path,
    model_name='evo-1.5-8k-base',
    output_dir='./evo_checkpoints/promoter_strength',
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    hidden_dim=512,
    dropout=0.2,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    device='cuda:0',
    local_model_path=None,
    use_similarity_split=True,
    similarity_threshold=0.8,
    kmer_size=3,
    resume_from=None,
    max_len=None,
    early_stopping_patience=20,
    task_type='regression',
    use_gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    test_size=0.15,
    val_size=0.2,
    n_folds=5,
    use_kfold_cv=False,
    use_log_label=False,
    use_neighbor_interp=False,
    interp_lambdas=None,
    use_mutation_augment=False,
    mutation_n=1,
    mutation_copies=1,
    # ---- Ablation experiments (fixed test set + precomputed fold assignment) ----
    evo_adaptation=None,
    pooling_mode='attention',
    fixed_test_seqs=None,
    fixed_test_labels=None,
    fold_assignment=None,
    single_split=False,
):
    """
    Fine-tune Evo for biopart prediction using LoRA.

    Args:
        data_path: Path to biopart dataset CSV (must have 'sequence' and 'activity' columns)
        model_name: Evo model name
        output_dir: Output directory for checkpoints
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension of regression head
        dropout: Dropout rate
        use_lora: Whether to use LoRA (default: True)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling factor (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.1)
        device: Device to use
        local_model_path: Path to local model directory (if not provided, will try to find in ./models/)
        use_similarity_split: Whether to use similarity-based cluster splitting for data split
        similarity_threshold: Maximum allowed similarity between train and val samples (0-1)
        kmer_size: k-mer size for similarity computation (default: 3)
        resume_from: Path to checkpoint to resume training from (optional)
        max_len: Maximum sequence length (if None, will auto-detect from data)
        early_stopping_patience: Number of epochs to wait for improvement before stopping (default: 20)
        task_type: 'regression' for promoter/rbs, 'classification' for terminator (default: 'regression')
        use_gradient_checkpointing: Whether to use gradient checkpointing to reduce memory (default: False)
        gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)
        test_size: Test set ratio (default: 0.15, i.e., 15% for test)
        val_size: Validation set ratio from remaining data (default: 0.2, i.e., 20% of remaining for val)
        n_folds: Number of folds for cross-validation (default: 5)
        use_kfold_cv: Whether to use K-fold cross-validation (default: False)
        use_log_label: Whether to apply log10(label+1) transform (for terminator, default: False)
        use_neighbor_interp: Whether to apply neighbor linear interpolation augmentation (default: False)
        interp_lambdas: Lambda values for interpolation (default: None → [0.5])
        use_mutation_augment: Whether to apply random mutation augmentation (default: False)
        mutation_n: Number of mutation positions per sequence (default: 1)
        mutation_copies: Number of mutated copies to generate (default: 1)

    Returns:
        If use_kfold_cv is True: (fold_model_paths, fold_val_scores, test_seqs, test_labels, max_len, fold_histories)
        If use_kfold_cv is False: None

    Data Split Strategy:
        If use_kfold_cv is True (5-fold cross-validation):
        - Test set: 15% (held out until final evaluation)
        - Remaining 85%: Used for 5-fold cross-validation
          * Each fold: 68% train / 17% val (from the 85%)
          * Each fold produces a separate model
          * Final prediction: Weighted average of all 5 fold models

        If use_kfold_cv is False (single split):
        - Train: 70% / Validation: 15% / Test: 15%
        - Single model with early stopping
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set checkpoint path for best model (fixed name for overwriting)
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    # Setup logging to file
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    # Redirect print to both console and log file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Open log file and setup tee
    log_handle = open(log_file, 'a', encoding='utf-8')
    sys.stdout = TeeOutput(sys.stdout, log_handle)
    sys.stderr = TeeOutput(sys.stderr, log_handle)
    
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Output directory: {output_dir}")
    print(f"Task type: {task_type}")
    print(f"{'='*60}")

    # Load data (normalize column names: Sequence→sequence, Label/Activity→activity)
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('sequence',):
            rename[c] = 'sequence'
        elif cl in ('activity', 'label'):
            rename[c] = 'activity'
    df.rename(columns=rename, inplace=True)
    sequences = df['sequence'].tolist()
    labels = df['activity'].values

    print(f"Total samples: {len(sequences)}")

    # Process labels based on task type
    if task_type == 'classification':
        # For classification, labels should be 0 or 1
        labels = labels.astype(np.int64)
        unique_labels = np.unique(labels)
        print(f"Classification task")
        print(f"Unique labels: {unique_labels}")
        print(f"Class distribution:")
        for label in unique_labels:
            count = np.sum(labels == label)
            print(f"  Class {label}: {count} ({count/len(labels)*100:.2f}%)")
    else:
        # For regression, use original labels directly
        labels = labels.astype(np.float32)
        print(f"Regression task")
        print(f"Strength range: [{labels.min():.4f}, {labels.max():.4f}]")

        # log10 label transform (regression tasks only)
        if use_log_label:
            labels = np.log10(labels + 1.0).astype(np.float32)
            import json as _json
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'label_transform.json'), 'w') as f:
                _json.dump({'transform': 'log10', 'shift': 1.0}, f, indent=2)
            print(f"[Label Transform] log10(label+1), range: [{labels.min():.4f}, {labels.max():.4f}]")
        else:
            print("Using raw label values")

    # Data augmentation (mutation + neighbor interpolation)
    if use_mutation_augment:
        orig_n = len(sequences)
        import random
        BASES = "ATGC"
        aug_seqs = []
        aug_labels = []
        for _ in range(mutation_copies):
            for seq, label in zip(sequences, labels):
                positions = random.sample(range(len(seq)), min(mutation_n, len(seq)))
                seq_list = list(seq)
                for pos in positions:
                    orig_base = seq_list[pos].upper()
                    new_base = random.choice([b for b in BASES if b != orig_base])
                    seq_list[pos] = new_base
                aug_seqs.append("".join(seq_list))
                aug_labels.append(label)
        sequences = sequences + aug_seqs
        labels = np.concatenate([labels, np.array(aug_labels, dtype=labels.dtype)])
        print(f"[Mutation Augment] {mutation_n} mutation(s) x {mutation_copies} copies, "
              f"samples {orig_n} -> {len(sequences)}")

    if use_neighbor_interp:
        # Find Hamming<=1 neighbor pairs and interpolate their labels (sequences are placeholders)
        if interp_lambdas is None:
            interp_lambdas = [0.5]
        BASES = "ATGC"
        seq_to_idx = {}
        for i, s in enumerate(sequences):
            s_upper = s.upper()
            if s_upper not in seq_to_idx:
                seq_to_idx[s_upper] = []
            seq_to_idx[s_upper].append(i)

        def _hamming1_variants(s):
            results = set()
            for pos in range(len(s)):
                orig = s[pos]
                for b in BASES:
                    if b != orig:
                        results.add(s[:pos] + b + s[pos + 1:])
            return results

        pairs = []
        seen = set()
        for i, s in enumerate(sequences):
            s_upper = s.upper()
            for nb in _hamming1_variants(s_upper):
                if nb in seq_to_idx:
                    for j in seq_to_idx[nb]:
                        if j > i and (i, j) not in seen:
                            seen.add((i, j))
                            pairs.append((i, j))

        n_pairs = len(pairs)
        n_interp = n_pairs * len(interp_lambdas)
        print(f"  [NeighborInterp] {n_pairs} Hamming<=1 neighbor pairs, "
              f"{len(interp_lambdas)} interpolation points -> +{n_interp} virtual samples")

        if n_interp > 0:
            aug_seqs = ["INTERP"] * n_interp
            aug_labels = np.zeros(n_interp, dtype=labels.dtype)
            idx = 0
            for lam in interp_lambdas:
                for i, j in pairs:
                    aug_labels[idx] = lam * labels[i] + (1 - lam) * labels[j]
                    idx += 1
            sequences = sequences + aug_seqs
            labels = np.concatenate([labels, aug_labels])
            print(f"[Neighbor Interp] {n_pairs} neighbor pairs x {len(interp_lambdas)} interpolation points, "
                  f"samples {len(sequences) - n_interp} -> {len(sequences)}")

    # Auto-detect sequence length statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"{'='*60}")
    print("Sequence Length Statistics:")
    print(f"{'='*60}")
    print(f"Min length:   {min(seq_lengths)}")
    print(f"Max length:   {max(seq_lengths)}")
    print(f"Mean length:  {np.mean(seq_lengths):.2f}")
    print(f"Median length: {np.median(seq_lengths):.2f}")
    print(f"25th percentile: {np.percentile(seq_lengths, 25):.2f}")
    print(f"75th percentile: {np.percentile(seq_lengths, 75):.2f}")
    print(f"90th percentile: {np.percentile(seq_lengths, 90):.2f}")
    print(f"95th percentile: {np.percentile(seq_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(seq_lengths, 99):.2f}")
    
    # Determine max_len
    if max_len is None:
        # Use 95th percentile as default max_len to cover most sequences while avoiding outliers
        max_len = int(np.percentile(seq_lengths, 95))
        print(f"\nAuto-detected max_len: {max_len} (95th percentile)")
    else:
        print(f"\nUsing specified max_len: {max_len}")
    
    # Warn if many sequences will be truncated
    truncated_count = sum(1 for length in seq_lengths if length > max_len)
    if truncated_count > 0:
        print(f"Warning: {truncated_count} sequences ({truncated_count/len(seq_lengths)*100:.2f}%) will be truncated to {max_len}")
    print(f"{'='*60}\n")

    # ============================================================================
    # Data Split Strategy
    # ============================================================================
    print(f"\n{'='*60}")
    print("Data Split Strategy")
    print(f"{'='*60}")
    print(f"Total samples: {len(sequences)}")
    print(f"use_kfold_cv: {use_kfold_cv}")
    print(f"use_similarity_split: {use_similarity_split}")

    if use_kfold_cv:
        print(f"n_folds: {n_folds}")
        print(f"K-Fold Cross-Validation mode:")
        print(f"  - Test set: {test_size*100}% (held out)")
        print(f"  - Remaining {(1-test_size)*100}%: {n_folds}-fold cross-validation")
    else:
        print(f"Single split mode:")
        print(f"  - Target: {int((1-test_size)*(1-val_size)*100)}% train / {int((1-test_size)*val_size*100)}% val / {int(test_size*100)}% test")

    # ============================================================================
    # Step 1: Split test set from full data
    # ============================================================================
    # Ablation mode: use the external fixed test set (passed in by the caller after overlap removal); no new random/cluster split
    use_fixed_test = fixed_test_seqs is not None
    if use_fixed_test:
        print(f"\n[Step 1/2] Using FIXED external test set ({len(fixed_test_seqs)} samples) — no re-split")
        test_seqs = list(fixed_test_seqs)
        test_labels = np.array(fixed_test_labels, dtype=labels.dtype)
        # trainval = all loaded training data (the caller should have excluded sequences overlapping the fixed test set)
        trainval_seqs = list(sequences)
        trainval_labels = np.array(labels, dtype=labels.dtype)
        trainval_labels_list = trainval_labels.tolist()
        test_labels_list = test_labels.tolist()
    elif use_similarity_split:
        print(f"\n[Step 1/2] Splitting test set ({test_size*100}% of data)...")
        print(f"  Using cluster-based splitting (similarity_threshold={similarity_threshold})")
        trainval_seqs, trainval_labels_list, test_seqs, test_labels_list = cluster_based_split(
            seqs=sequences,
            labels=labels.tolist() if task_type == 'classification' else labels.tolist(),
            test_size=test_size,
            similarity_threshold=similarity_threshold,
            kmer_size=kmer_size,
            random_state=42,
            verbose=True
        )
    else:
        print(f"\n[Step 1/2] Splitting test set ({test_size*100}% of data)...")
        print(f"  Using random split")
        trainval_seqs, test_seqs, trainval_labels, test_labels = train_test_split(
            sequences, labels, test_size=test_size, random_state=42
        )
        trainval_labels_list = trainval_labels.tolist()
        test_labels_list = test_labels.tolist()

    trainval_labels = np.array(trainval_labels_list, dtype=labels.dtype)
    test_labels = np.array(test_labels_list, dtype=labels.dtype)

    print(f"\n  Train+Val size: {len(trainval_seqs)} ({len(trainval_seqs)/len(sequences)*100:.1f}%)")
    print(f"  Test size: {len(test_seqs)} ({len(test_seqs)/len(sequences)*100:.1f}%)")

    # ============================================================================
    # Step 2: Prepare for K-Fold Cross-Validation or Single Split
    # ============================================================================
    print(f"\n[Step 2/2] Preparing for training...")

    if use_kfold_cv:
        # Use K-Fold Cross-Validation
        print(f"\n  Using {n_folds}-Fold Cross-Validation")
        print(f"  use_similarity_split: {use_similarity_split}")

        # Create K-Fold splitter
        if fold_assignment is not None:
            # Ablation mode: use the precomputed fold assignment (from the manifest); no re-clustering
            print(f"  Using PRECOMPUTED fold assignment from manifest (no re-cluster)")
            kf = None
        elif use_similarity_split:
            print(f"  Cluster-based K-Fold (similarity_threshold={similarity_threshold})")
            kf = ClusterBasedKFold(
                n_splits=n_folds,
                similarity_threshold=similarity_threshold,
                kmer_size=kmer_size,
                random_state=42,
                verbose=True
            )
            # Fit the cluster-based splitter
            kf._fit(trainval_seqs)
        else:
            print(f"  Random K-Fold")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        trainval_labels_np = np.array(trainval_labels_list, dtype=labels.dtype)

        # Store for K-Fold training
        fold_data = {
            'sequences': trainval_seqs,
            'labels': trainval_labels_np,
            'kf': kf,
            'use_similarity_split': use_similarity_split,
            'fold_assignment': fold_assignment,
        }
    else:
        # Single split (not K-Fold)
        print(f"\n  Using single train/val split")
        adjusted_val_size = val_size / (1 - test_size)

        if use_similarity_split:
            print(f"  Cluster-based split (similarity_threshold={similarity_threshold})")
            train_seqs, train_labels_list, val_seqs, val_labels_list = cluster_based_split(
                seqs=trainval_seqs,
                labels=trainval_labels.tolist() if task_type == 'classification' else trainval_labels.tolist(),
                test_size=adjusted_val_size,
                similarity_threshold=similarity_threshold,
                kmer_size=kmer_size,
                random_state=42,
                verbose=True
            )
            train_labels = np.array(train_labels_list, dtype=labels.dtype)
            val_labels = np.array(val_labels_list, dtype=labels.dtype)
        else:
            print(f"  Random split")
            train_seqs, val_seqs, train_labels, val_labels = train_test_split(
                trainval_seqs, trainval_labels, test_size=adjusted_val_size, random_state=42
            )

        print(f"\n  Final split:")
        print(f"    Train: {len(train_seqs)} ({len(train_seqs)/len(sequences)*100:.1f}%)")
        print(f"    Val:   {len(val_seqs)} ({len(val_seqs)/len(sequences)*100:.1f}%)")
        print(f"    Test:  {len(test_seqs)} ({len(test_seqs)/len(sequences)*100:.1f}%)")

        fold_data = None

    print(f"{'='*60}\n")

    # Load Evo model
    # Check if local model path is provided or exists in default location
    if local_model_path is None:
        # Try to find local model in ./models/ directory
        default_local_path = os.path.join(os.path.dirname(__file__), 'models', model_name)
        if os.path.exists(default_local_path):
            local_model_path = default_local_path
            print(f"Found local model at: {local_model_path}")

    if local_model_path and os.path.exists(local_model_path):
        evo_model = load_evo_from_local(local_model_path, model_name, device)
    else:
        print(f"Loading Evo model from HuggingFace: {model_name}")
        evo_model = Evo(model_name)
        evo_model.model.to(device)

    # ============================================================================
    # K-Fold Cross-Validation Training
    # ============================================================================
    if use_kfold_cv:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Training")
        print(f"{'='*60}")
        print(f"n_folds: {n_folds}")
        print(f"Train+Val samples: {len(trainval_seqs)}")

        # Store fold metrics
        fold_model_paths = []
        fold_val_scores = []
        fold_histories = []

        # Get K-Fold splits
        kf = fold_data['kf']
        sequences = fold_data['sequences']
        labels = fold_data['labels']
        precomp_assignment = fold_data.get('fold_assignment')

        # Build (train_idx, val_idx) per fold: prefer the precomputed manifest split
        if precomp_assignment is not None:
            precomp_assignment = np.asarray(precomp_assignment)
            if single_split:
                # No K-fold CV: use fold==1 as the validation set and train a single model (5x speed-up)
                fold_splits = [(np.where(precomp_assignment != 1)[0],
                                np.where(precomp_assignment == 1)[0])]
            else:
                fold_splits = []
                for k in range(1, n_folds + 1):
                    val_idx = np.where(precomp_assignment == k)[0]
                    train_idx = np.where(precomp_assignment != k)[0]
                    fold_splits.append((train_idx, val_idx))
            split_iter = enumerate(fold_splits, 1)
        else:
            split_iter = enumerate(kf.split(sequences, labels), 1)

        for fold_idx, (train_idx, val_idx) in split_iter:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}/{n_folds}")
            print(f"{'='*60}")

            # Get train and val data for this fold
            fold_train_seqs = [sequences[i] for i in train_idx]
            fold_val_seqs = [sequences[i] for i in val_idx]
            fold_train_labels = labels[train_idx]
            fold_val_labels = labels[val_idx]

            print(f"Train: {len(fold_train_seqs)}, Val: {len(fold_val_seqs)}")

            # Create fold checkpoint directory
            fold_checkpoint_dir = os.path.join(output_dir, f'fold_{fold_idx}')
            os.makedirs(fold_checkpoint_dir, exist_ok=True)

            # Train this fold
            _fold_start_time = time.perf_counter()
            fold_model_path, fold_val_score, fold_history = train_single_fold(
                fold_idx=fold_idx,
                train_seqs=fold_train_seqs,
                train_labels=fold_train_labels,
                val_seqs=fold_val_seqs,
                val_labels=fold_val_labels,
                evo_model=evo_model,
                fold_checkpoint_dir=fold_checkpoint_dir,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                device=device,
                max_len=max_len,
                task_type=task_type,
                use_gradient_checkpointing=use_gradient_checkpointing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience,
                batch_size=batch_size,
                use_log_label=use_log_label,
                evo_adaptation=evo_adaptation,
                pooling_mode=pooling_mode
            )
            _fold_runtime = time.perf_counter() - _fold_start_time

            # Ablation mode: write per-fold detailed metrics to fold_details.jsonl
            # (consistent with the CNN–Attention–BiLSTM path; consumed by the aggregator)
            if use_fixed_test:
                _record = {'fold': fold_idx, 'split': 'val', 'runtime_sec': _fold_runtime}
                if fold_history:
                    if task_type == 'classification':
                        _scores = fold_history.get('val_f1', fold_history.get('val_accuracy', []))
                    else:
                        _scores = fold_history.get('val_pearson', [])
                    if _scores:
                        _best_idx = int(np.nanargmax(_scores))
                        _record['best_epoch'] = _best_idx
                        if task_type == 'classification':
                            for _k in ('accuracy', 'precision', 'recall', 'f1'):
                                _vals = fold_history.get(f'val_{_k}', [])
                                if _vals:
                                    _record[_k] = _vals[_best_idx]
                        else:
                            for _k in ('pearson', 'spearman', 'mse', 'mae', 'r2'):
                                _vals = fold_history.get(f'val_{_k}', [])
                                if _vals:
                                    _record[{'pearson': 'pearson_r', 'spearman': 'spearman_r',
                                             'mse': 'mse', 'mae': 'mae', 'r2': 'r2'}[_k]] = _vals[_best_idx]
                _fd_path = os.path.join(output_dir, 'fold_details.jsonl')
                with open(_fd_path, 'a') as _f:
                    _f.write(json.dumps(_record, default=float) + '\n')

            fold_model_paths.append(fold_model_path)
            fold_val_scores.append(fold_val_score)
            fold_histories.append(fold_history)

        # Calculate and print cross-validation results
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Results")
        print(f"{'='*60}")

        if task_type == 'classification':
            for i, score in enumerate(fold_val_scores, 1):
                print(f"Fold {i} F1: {score:.6f}")
            mean_f1 = np.mean(fold_val_scores)
            std_f1 = np.std(fold_val_scores)
            print(f"\nMean F1: {mean_f1:.6f} ± {std_f1:.6f}")
            best_metric = mean_f1
        else:
            for i, score in enumerate(fold_val_scores, 1):
                print(f"Fold {i} Pearson r: {score:.6f}")
            mean_r2 = np.mean(fold_val_scores)
            std_r2 = np.std(fold_val_scores)
            print(f"\nMean Pearson r: {mean_r2:.6f} ± {std_r2:.6f}")
            best_metric = mean_r2

        # Save CV results summary
        cv_summary_path = os.path.join(output_dir, 'cv_summary.txt')
        with open(cv_summary_path, 'w') as f:
            f.write(f"K-Fold Cross-Validation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"n_folds: {n_folds}\n")
            f.write(f"Task type: {task_type}\n")
            f.write(f"use_similarity_split: {use_similarity_split}\n")
            f.write(f"similarity_threshold: {similarity_threshold}\n")
            f.write(f"kmer_size: {kmer_size}\n")
            f.write(f"Train+Val samples: {len(trainval_seqs)}\n")
            f.write(f"Test samples: {len(test_seqs)}\n\n")

            if task_type == 'classification':
                f.write("Fold F1 Scores:\n")
                for i, score in enumerate(fold_val_scores, 1):
                    f.write(f"  Fold {i}: {score:.6f}\n")
                f.write(f"\nMean F1: {mean_f1:.6f} ± {std_f1:.6f}\n")
            else:
                f.write("Fold Pearson r Scores:\n")
                for i, score in enumerate(fold_val_scores, 1):
                    f.write(f"  Fold {i}: {score:.6f}\n")
                f.write(f"\nMean Pearson r: {mean_r2:.6f} ± {std_r2:.6f}\n")

            f.write(f"\nFold Model Paths:\n")
            for i, path in enumerate(fold_model_paths, 1):
                f.write(f"  Fold {i}: {path}\n")

        print(f"\nCV summary saved to: {cv_summary_path}")

        # Save test set data
        if test_seqs is not None and len(test_seqs) > 0:
            test_data_path = os.path.join(output_dir, 'test_data.npz')
            np.savez_compressed(
                test_data_path,
                sequences=np.array(test_seqs, dtype=object),
                labels=test_labels,
                max_len=max_len,
                fold_model_paths=np.array(fold_model_paths, dtype=object),
                fold_val_scores=np.array(fold_val_scores)
            )
            print(f"Test data saved to: {test_data_path}")
            print(f"Fold model paths: {len(fold_model_paths)}")

        # Clean up evo_model to free GPU memory before testing
        print("Cleaning up evo_model to free GPU memory...")
        del evo_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory cleanup completed.")

        return fold_model_paths, fold_val_scores, test_seqs, test_labels, max_len, fold_histories

    else:
        # ============================================================================
        # Single Split Training (original behavior, no K-Fold CV)
        # ============================================================================
        print(f"\n{'='*60}")
        print(f"Single Split Training (No K-Fold CV)")
        print(f"{'='*60}")

        # Create model with appropriate head
        model = EvoWithRegressionHead(
            evo_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=task_type,
            use_gradient_checkpointing=use_gradient_checkpointing,
            evo_adaptation=evo_adaptation,
            pooling_mode=pooling_mode
        ).to(device)

        # Create datasets and dataloaders
        train_dataset = PromoterDataset(train_seqs, train_labels, evo_model.tokenizer,
                                         max_len=max_len, task_type=task_type, augment=True)
        val_dataset = PromoterDataset(val_seqs, val_labels, evo_model.tokenizer,
                                       max_len=max_len, task_type=task_type, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler with warmup and cosine annealing
        total_steps = len(train_loader) * num_epochs
        warmup_steps = len(train_loader) * 2  # 2 epochs warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        print(f"Using Cosine Annealing with Warmup: {warmup_steps} warmup steps, {total_steps} total steps")

        # Loss function
        if task_type == 'classification':
            # Compute class weights to handle imbalance
            label_counts = np.bincount(train_labels)
            class_weights = 1.0 / (label_counts / label_counts.sum())
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"Class distribution: {label_counts.tolist()}")
            print(f"Class weights: {class_weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted CrossEntropyLoss for classification")
        else:
            criterion = nn.MSELoss()
            print("Using MSELoss for regression")

        # NOTE: GradScaler is NOT used for BFloat16 models
        # BFloat16 has the same exponent range as FP32, so it doesn't need gradient scaling
        # GradScaler is only needed for FP16 (float16) which has limited numeric range
        scaler = None
        print("Training with BFloat16 precision (GradScaler not needed/used)")

        # Training loop
        best_val_loss = float('inf')
        best_metric = -float('inf')
        epochs_no_improve = 0

        # Initialize history lists for tracking training metrics
        train_losses = []
        val_losses = []
        if task_type == 'classification':
            val_accuracies = []
            val_precisions = []
            val_recalls = []
            val_f1s = []
        else:  # regression
            val_mses = []
            val_maes = []
            val_r2s = []
            val_pearsons = []
            val_spearmans = []

        # Set checkpoint path for best model
        best_model_path = os.path.join(output_dir, 'best_model.pth')

        # Print header for epoch progress
        print(f"\n{'='*60}")
        print(f"Single Split Training - Progress")
        print(f"{'='*60}")
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12}", end="")
        if task_type == 'classification':
            print(f" | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Status':>15}")
        else:
            print(f" | {'MSE':>12} | {'MAE':>12} | {'R²':>10} | {'Pearson':>10} | {'Spearman':>10} | {'Status':>15}")
        print(f"{'='*160}")

        for epoch in range(1, num_epochs + 1):
            # Train
            # FIX #2: Pass scaler to train_epoch for AMP support
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                    scaler, task_type, gradient_accumulation_steps)

            # Validate
            val_results = validate(model, val_loader, criterion, device, task_type)
            val_loss = val_results[0]

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Get metric based on task type and prepare status message
            status_msg = ""

            if task_type == 'classification':
                # val_results: (val_loss, accuracy, precision, recall, f1)
                val_accuracy = val_results[1]
                val_precision = val_results[2]
                val_recall = val_results[3]
                val_f1 = val_results[4]
                val_metric = val_f1  # F1 score is the main metric

                # Early stopping check
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    status_msg = "*** BEST MODEL SAVED ***"

                    # Save model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metric': val_metric,
                        'epoch': epoch
                    }, best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        status_msg = f"EARLY STOPPING (no improvement for {epochs_no_improve} epochs)"

                # Record history
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                val_precisions.append(val_precision)
                val_recalls.append(val_recall)
                val_f1s.append(val_f1)

                # Print epoch results
                print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                      f"{val_accuracy:>10.6f} | {val_precision:>10.6f} | {val_recall:>10.6f} | "
                      f"{val_f1:>10.6f} | {status_msg:>15}")

            else:  # regression
                # val_results: (val_loss, mse, mae, r2, pearson_corr, spearman_corr)
                val_mse = val_results[1]
                val_mae = val_results[2]
                val_r2 = val_results[3]
                val_pearson = val_results[4]
                val_spearman = val_results[5]
                val_metric = val_pearson  # Use Pearson correlation as the main metric for early stopping

                # Early stopping check (based on Pearson correlation)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    status_msg = "*** BEST MODEL SAVED ***"

                    # Save model
                    checkpoint_data = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metric': val_metric,
                        'val_r2': val_r2,
                        'val_pearson': val_pearson,
                        'val_spearman': val_spearman,
                        'epoch': epoch
                    }
                    if use_log_label:
                        checkpoint_data['label_transform'] = 'log10'
                    torch.save(checkpoint_data, best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        status_msg = f"EARLY STOPPING (no improvement for {epochs_no_improve} epochs)"

                # Record history
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_mses.append(val_mse)
                val_maes.append(val_mae)
                val_r2s.append(val_r2)
                val_pearsons.append(val_pearson)
                val_spearmans.append(val_spearman)

                # Print epoch results
                print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                      f"{val_mse:>12.6f} | {val_mae:>12.6f} | {val_r2:>10.6f} | "
                      f"{val_pearson:>10.6f} | {val_spearman:>10.6f} | {status_msg:>15}")

            # Learning rate scheduling (step after each epoch for cosine scheduler)
            scheduler.step()

            # Print learning rate if changed
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"         Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")
                current_lr = new_lr

            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best {'F1' if task_type == 'classification' else 'Pearson'}: {best_metric:.6f} (epoch {epoch - epochs_no_improve})")
                print(f"{'='*60}")
                break

        # Print training completion summary
        print(f"\n{'='*60}")
        print(f"Training Completed")
        print(f"{'='*60}")
        if task_type == 'classification':
            print(f"Best validation F1: {best_metric:.6f}")
        else:
            print(f"Best validation Pearson: {best_metric:.6f} (early stopping metric)")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Model saved to: {best_model_path}")
        print(f"{'='*60}\n")

        # Create history dictionary and plot training curves
        if task_type == 'classification':
            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_accuracy': val_accuracies,
                'val_precision': val_precisions,
                'val_recall': val_recalls,
                'val_f1': val_f1s
            }
        else:  # regression
            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_mse': val_mses,
                'val_mae': val_maes,
                'val_r2': val_r2s,
                'val_pearson': val_pearsons,
                'val_spearman': val_spearmans
            }

        # Plot and save training history
        print("\nGenerating training history plots...")
        plot_training_history(history, output_dir, task_type)

        # Save training history to file for later analysis
        history_path = os.path.join(output_dir, 'training_history.npz')
        np.savez_compressed(history_path, **history)
        print(f"Training history saved to: {history_path}")

        # Save test set data
        if test_seqs is not None and len(test_seqs) > 0:
            test_data_path = os.path.join(output_dir, 'test_data.npz')
            np.savez_compressed(
                test_data_path,
                sequences=np.array(test_seqs, dtype=object),
                labels=test_labels,
                max_len=max_len
            )
            print(f"Test data saved to: {test_data_path}")

        # Clean up training model and optimizer to free GPU memory before testing
        print("Cleaning up training model to free GPU memory...")
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory cleanup completed.")

        # Return single model path (not a list) for consistency with test_model
        return [best_model_path], [best_metric], test_seqs, test_labels, max_len



def plot_training_history(history, output_dir, task_type='regression'):
    """
    Plot validation metrics over epochs.

    Args:
        history: Dictionary containing training history
        output_dir: Directory to save the plot
        task_type: 'regression' or 'classification'
    """
    if task_type == 'classification':
        epochs = range(1, len(history['val_f1']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History (Classification)', fontsize=16, fontweight='bold')

        # Train/Val Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train/Val Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Val Accuracy
        axes[0, 1].plot(epochs, history['val_accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].grid(True, alpha=0.3)

        # Val Precision
        axes[0, 2].plot(epochs, history['val_precision'], 'm-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Validation Precision')
        axes[0, 2].grid(True, alpha=0.3)

        # Val Recall
        axes[1, 0].plot(epochs, history['val_recall'], 'c-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Validation Recall')
        axes[1, 0].grid(True, alpha=0.3)

        # Val F1
        axes[1, 1].plot(epochs, history['val_f1'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].grid(True, alpha=0.3)

        # Hide last subplot
        axes[1, 2].axis('off')
    else:
        epochs = range(1, len(history['val_r2']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History (Regression)', fontsize=16, fontweight='bold')

        # Train/Val Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train/Val Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Val MSE
        axes[0, 1].plot(epochs, history['val_mse'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('Validation MSE')
        axes[0, 1].grid(True, alpha=0.3)

        # Val MAE
        axes[0, 2].plot(epochs, history['val_mae'], 'm-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].set_title('Validation MAE')
        axes[0, 2].grid(True, alpha=0.3)

        # Val R²
        axes[1, 0].plot(epochs, history['val_r2'], 'c-', linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Validation R²')
        axes[1, 0].grid(True, alpha=0.3)

        # Val Pearson
        axes[1, 1].plot(epochs, history['val_pearson'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pearson Correlation')
        axes[1, 1].set_title('Validation Pearson')
        axes[1, 1].grid(True, alpha=0.3)

        # Val Spearman
        axes[1, 2].plot(epochs, history['val_spearman'], 'purple', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Spearman Correlation')
        axes[1, 2].set_title('Validation Spearman')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {plot_path}")

    plt.close()


def train_single_fold(fold_idx, train_seqs, train_labels, val_seqs, val_labels,
                       evo_model, fold_checkpoint_dir, num_epochs, learning_rate,
                       hidden_dim, dropout, use_lora, lora_r, lora_alpha, lora_dropout,
                       device, max_len, task_type, use_gradient_checkpointing,
                       gradient_accumulation_steps, early_stopping_patience, batch_size,
                       use_log_label=False,
                       evo_adaptation=None, pooling_mode='attention'):
    """
    Train a single fold for K-fold cross-validation.

    Args:
        use_log_label: Whether log10(label+1) transform was applied to labels
        evo_adaptation: 'lora' | 'head_only' | 'partial_ft' (None -> inferred from use_lora)
        pooling_mode: 'attention' | 'mean'

    Returns:
        fold_model_path: Path to the saved fold model
        fold_val_score: Best validation score for this fold
        fold_history: Dictionary containing training metrics history for this fold
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_seqs)}, Val samples: {len(val_seqs)}")

    # Create fold-specific model
    fold_model = EvoWithRegressionHead(
        evo_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=task_type,
        use_gradient_checkpointing=use_gradient_checkpointing,
        evo_adaptation=evo_adaptation,
        pooling_mode=pooling_mode
    ).to(device)

    # Create datasets and dataloaders
    train_dataset = PromoterDataset(train_seqs, train_labels, evo_model.tokenizer,
                                     max_len=max_len, task_type=task_type, augment=True)
    val_dataset = PromoterDataset(val_seqs, val_labels, evo_model.tokenizer,
                                   max_len=max_len, task_type=task_type, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    trainable_params = [p for p in fold_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup and cosine annealing
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Loss function
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # NOTE: GradScaler is NOT used for BFloat16 models
    # BFloat16 has the same exponent range as FP32, so it doesn't need gradient scaling
    scaler = None

    # Training loop for this fold
    best_val_loss = float('inf')
    best_metric = -float('inf')
    epochs_no_improve = 0

    # Initialize history lists for tracking training metrics
    train_losses = []
    val_losses = []
    if task_type == 'classification':
        val_accuracies = []
        val_precisions = []
        val_recalls = []
        val_f1s = []
    else:  # regression
        val_mses = []
        val_maes = []
        val_r2s = []
        val_pearsons = []
        val_spearmans = []

    fold_model_path = os.path.join(fold_checkpoint_dir, 'checkpoint.pth')

    # Print header for epoch progress
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} - Training Progress")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12}", end="")
    if task_type == 'classification':
        print(f" | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Status':>15}")
    else:
        print(f" | {'MSE':>12} | {'MAE':>12} | {'R²':>10} | {'Pearson':>10} | {'Spearman':>10} | {'Status':>15}")
    print(f"{'='*160}")

    for epoch in range(1, num_epochs + 1):
        # Train
        # FIX #2: Pass scaler to train_epoch for AMP support
        train_loss = train_epoch(fold_model, train_loader, optimizer, criterion, device,
                                scaler, task_type, gradient_accumulation_steps)

        # Validate
        val_results = validate(fold_model, val_loader, criterion, device, task_type)
        val_loss = val_results[0]

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Get metric based on task type and prepare status message
        status_msg = ""
        is_best = False

        if task_type == 'classification':
            # val_results: (val_loss, accuracy, precision, recall, f1)
            val_accuracy = val_results[1]
            val_precision = val_results[2]
            val_recall = val_results[3]
            val_f1 = val_results[4]
            val_metric = val_f1  # F1 score is the main metric

            # Early stopping check
            if val_metric > best_metric:
                best_metric = val_metric
                best_val_loss = val_loss
                epochs_no_improve = 0
                is_best = True
                status_msg = "*** BEST MODEL SAVED ***"

                # Save fold model
                checkpoint_data = {
                    'model_state_dict': fold_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'epoch': epoch
                }
                if use_log_label:
                    checkpoint_data['label_transform'] = 'log10'
                checkpoint_data['evo_adaptation'] = fold_model.evo_adaptation
                checkpoint_data['pooling_mode'] = fold_model.pooling_mode
                torch.save(checkpoint_data, fold_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    status_msg = f"EARLY STOPPING (no improvement for {epochs_no_improve} epochs)"

            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1s.append(val_f1)

            # Print epoch results
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                  f"{val_accuracy:>10.6f} | {val_precision:>10.6f} | {val_recall:>10.6f} | "
                  f"{val_f1:>10.6f} | {status_msg:>15}")

        else:  # regression
            # val_results: (val_loss, mse, mae, r2, pearson_corr, spearman_corr)
            val_mse = val_results[1]
            val_mae = val_results[2]
            val_r2 = val_results[3]
            val_pearson = val_results[4]
            val_spearman = val_results[5]
            val_metric = val_pearson  # Use Pearson correlation as the main metric for early stopping

            # Early stopping check (based on Pearson correlation)
            if val_metric > best_metric:
                best_metric = val_metric
                best_val_loss = val_loss
                epochs_no_improve = 0
                is_best = True
                status_msg = "*** BEST MODEL SAVED ***"

                # Save fold model
                checkpoint_data = {
                    'model_state_dict': fold_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'val_r2': val_r2,
                    'val_pearson': val_pearson,
                    'val_spearman': val_spearman,
                    'epoch': epoch
                }
                if use_log_label:
                    checkpoint_data['label_transform'] = 'log10'
                checkpoint_data['evo_adaptation'] = fold_model.evo_adaptation
                checkpoint_data['pooling_mode'] = fold_model.pooling_mode
                torch.save(checkpoint_data, fold_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    status_msg = f"EARLY STOPPING (no improvement for {epochs_no_improve} epochs)"

            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mses.append(val_mse)
            val_maes.append(val_mae)
            val_r2s.append(val_r2)
            val_pearsons.append(val_pearson)
            val_spearmans.append(val_spearman)

            # Print epoch results
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                  f"{val_mse:>12.6f} | {val_mae:>12.6f} | {val_r2:>10.6f} | "
                  f"{val_pearson:>10.6f} | {val_spearman:>10.6f} | {status_msg:>15}")

        # Learning rate scheduling (step after each epoch for cosine scheduler)
        scheduler.step()

        # Print learning rate if changed
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"         Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")
            current_lr = new_lr

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}: Early stopping triggered at epoch {epoch}")
            print(f"Best {'F1' if task_type == 'classification' else 'Pearson'}: {best_metric:.6f} (epoch {epoch - epochs_no_improve})")
            print(f"{'='*60}")
            break

    # Print fold completion summary
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} Training Completed")
    print(f"{'='*60}")
    if task_type == 'classification':
        print(f"Best validation F1: {best_metric:.6f}")
    else:
        print(f"Best validation Pearson: {best_metric:.6f} (early stopping metric)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {fold_model_path}")
    print(f"{'='*60}\n")

    # Create history dictionary and plot training curves for this fold
    if task_type == 'classification':
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies,
            'val_precision': val_precisions,
            'val_recall': val_recalls,
            'val_f1': val_f1s
        }
    else:  # regression
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_mse': val_mses,
            'val_mae': val_maes,
            'val_r2': val_r2s,
            'val_pearson': val_pearsons,
            'val_spearman': val_spearmans
        }

    # Plot and save training history for this fold
    print(f"Generating training history plots for Fold {fold_idx}...")
    plot_training_history(history, fold_checkpoint_dir, task_type)

    # Save training history to file for later analysis
    history_path = os.path.join(fold_checkpoint_dir, f'fold_{fold_idx}_training_history.npz')
    np.savez_compressed(history_path, **history)
    print(f"Fold {fold_idx} training history saved to: {history_path}")

    # Return fold history for later plotting
    if task_type == 'classification':
        fold_history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies,
            'val_precision': val_precisions,
            'val_recall': val_recalls,
            'val_f1': val_f1s
        }
    else:  # regression
        fold_history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_mse': val_mses,
            'val_mae': val_maes,
            'val_r2': val_r2s,
            'val_pearson': val_pearsons,
            'val_spearman': val_spearmans
        }

    return fold_model_path, best_metric, fold_history


def ensemble_predict(model_paths, test_seqs, test_labels, tokenizer,
                     device, batch_size, task_type, max_len, ensemble_method='weighted'):
    """
    Ensemble prediction from multiple fold models.

    Args:
        model_paths: List of paths to fold model checkpoints
        test_seqs: Test sequences
        test_labels: Test labels
        tokenizer: Evo tokenizer
        device: Device to use
        batch_size: Batch size
        task_type: 'regression' or 'classification'
        max_len: Maximum sequence length
        ensemble_method: 'weighted' or 'mean'

    Returns:
        predictions: Ensemble predictions
    """
    print(f"\n{'='*60}")
    print(f"Ensemble Prediction ({len(model_paths)} models)")
    print(f"Method: {ensemble_method}")
    print(f"{'='*60}")

    all_fold_predictions = []

    for fold_idx, model_path in enumerate(model_paths, 1):
        print(f"Loading fold {fold_idx} model...")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Get fold-specific metrics (for weighted averaging)
        fold_val_metric = checkpoint.get('val_metric', 0.0)

        # Create model and load weights
        fold_model = EvoWithRegressionHead(
            evo_model=None,  # Will be replaced
            hidden_dim=512,
            dropout=0.2,
            use_lora=True,
            task_type=task_type
        ).to(device)

        # Load state dict
        state_dict = checkpoint['model_state_dict']
        fold_model.load_state_dict(state_dict)

        fold_model.eval()

        # Create test dataset
        test_dataset = PromoterDataset(test_seqs, test_labels, tokenizer,
                                       max_len=max_len, task_type=task_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        fold_predictions = []
        with torch.no_grad():
            for input_ids, _, attention_mask in test_loader:
                input_ids = input_ids.to(device)
                preds = fold_model(input_ids)

                if task_type == 'classification':
                    fold_predictions.append(preds.cpu().numpy())
                else:
                    fold_predictions.extend(preds.cpu().numpy())

        if task_type == 'classification':
            fold_predictions = np.concatenate(fold_predictions, axis=0)
        else:
            fold_predictions = np.array(fold_predictions)

        all_fold_predictions.append(fold_predictions)
        print(f"  Fold {fold_idx}: val_metric={fold_val_metric:.4f}")

        del fold_model
        torch.cuda.empty_cache()

    # Ensemble predictions
    all_fold_predictions = np.array(all_fold_predictions)

    if task_type == 'classification':
        # For classification: average logits and get class
        avg_logits = np.mean(all_fold_predictions, axis=0)
        predictions = np.argmax(avg_logits, axis=1)
    else:
        # For regression: average predictions
        if ensemble_method == 'weighted':
            # Use validation scores as weights
            weights = np.array([torch.load(p, map_location='cpu').get('val_metric', 1.0)
                              for p in model_paths])
            weights = weights / np.sum(weights)
            predictions = np.average(all_fold_predictions, axis=0, weights=weights)
            print(f"Weights: {weights}")
        else:
            predictions = np.mean(all_fold_predictions, axis=0)

    return predictions


def predict(model, sequences, tokenizer, device='cuda:0', batch_size=32, task_type='regression'):
    """
    Predict using fine-tuned model.

    FIX #1: Now handles attention_mask from dataset to prevent padding leakage.

    Args:
        model: Fine-tuned model
        sequences: List of DNA sequences
        tokenizer: Evo tokenizer
        device: Device to use
        batch_size: Batch size for prediction
        task_type: 'regression' or 'classification'

    Returns:
        predictions: Array of predicted values (strengths for regression, class probabilities/indices for classification)
    """
    model.eval()
    predictions = []

    # Create dataset (dummy labels are fine since we only need predictions)
    dataset = PromoterDataset(sequences, [0]*len(sequences), tokenizer, task_type=task_type)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_data in dataloader:
            # FIX #1: Unpack attention_mask along with input_ids
            if len(batch_data) == 3:
                input_ids, _, attention_mask = batch_data
            else:
                # Backward compatibility
                input_ids, _ = batch_data
                attention_mask = None

            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # FIX #1: Pass attention_mask to model
            preds = model(input_ids, attention_mask)

            if task_type == 'classification':
                # Get predicted class and probabilities
                probs = torch.softmax(preds, dim=1)
                pred_classes = torch.argmax(preds, dim=1)
                predictions.extend(pred_classes.cpu().numpy())
            else:
                predictions.extend(preds.cpu().numpy())

    predictions = np.array(predictions)

    return predictions


def test_model(model, test_seqs, test_labels, tokenizer, device='cuda:0',
               batch_size=32, task_type='regression', output_dir=None, max_len=None,
               fold_model_paths=None, fold_val_scores=None, ensemble_method='weighted',
               evo_model=None, local_model_path=None, model_name='evo-1.5-8k-base',
               checkpoint_path=None):
    """
    Evaluate the model on test set.

    Args:
        model: Trained model (for single model evaluation) or None (for ensemble)
        test_seqs: Test set sequences
        test_labels: Test set labels
        tokenizer: Evo tokenizer
        device: Device to use
        batch_size: Batch size for evaluation
        task_type: 'regression' or 'classification'
        output_dir: Directory to save test results
        max_len: Maximum sequence length
        fold_model_paths: List of fold model paths for ensemble prediction
        fold_val_scores: List of validation scores for weighted ensemble
        ensemble_method: 'mean', 'weighted', or 'median' for ensemble
        evo_model: Evo model wrapper (needed for ensemble prediction)
        checkpoint_path: Path to checkpoint for single model (needed for label scaler)

    Returns:
        test_metrics: Dictionary of test metrics
    """
    import csv

    # Determine if using ensemble prediction
    use_ensemble = fold_model_paths is not None and len(fold_model_paths) > 0

    if use_ensemble:
        print(f"\n{'='*70}")
        print(f"Ensemble prediction using {len(fold_model_paths)} fold models")
        print(f"Task type: {task_type}")
        print(f"Ensemble method: {ensemble_method}")
        print(f"{'='*70}")

        # Create test dataset
        test_dataset = PromoterDataset(test_seqs, test_labels, tokenizer, max_len=max_len,
                                       task_type=task_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_fold_predictions = []
        all_targets_original = []

        # Get all targets
        for _, labels, _ in test_loader:
            all_targets_original.extend(labels.cpu().numpy())

        all_targets_original = np.array(all_targets_original)

        # Get predictions from each fold model
        # FIX #4: Memory-efficient ensemble loading - load base Evo ONCE and reuse for all folds

        # First, load the first checkpoint to detect architecture and prepare shared Evo model
        print(f"Loading the first fold checkpoint to detect the architecture...")

        # Import model class
        current_dir = os.path.dirname(os.path.abspath(__file__))
        import importlib.util
        spec = importlib.util.spec_from_file_location("lora_finetune_evo", os.path.join(current_dir, 'lora_finetune_evo.py'))
        finetune_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finetune_module)

        # Load first checkpoint to detect architecture
        first_checkpoint = torch.load(fold_model_paths[0], map_location='cpu', weights_only=False)
        first_state_dict = first_checkpoint['model_state_dict'] if isinstance(first_checkpoint, dict) and 'model_state_dict' in first_checkpoint else None

        if first_state_dict is None:
            raise ValueError("Expected checkpoint dict with 'model_state_dict' key")

        # Detect architecture from state dict
        hidden_dim = None
        for key in first_state_dict.keys():
            if 'head.regression.0.weight' in key:
                hidden_dim = first_state_dict[key].shape[0]
                break

        lora_r = None
        for key in first_state_dict.keys():
            if 'lora_A' in key and 'blocks.0' in key:
                lora_r = first_state_dict[key].shape[0]
                break

        lora_alpha = lora_r * 2 if lora_r else 32

        # FIX #4: Load the base Evo model ONCE and keep it in memory
        # This massive model is shared across all folds - only the LoRA matrices and head differ
        if local_model_path and model_name:
            print(f"  Loading the shared base Evo model (loaded once only)...")
            shared_evo_wrapper = finetune_module.load_evo_from_local(local_model_path, model_name, 'cpu')
        elif evo_model is not None:
            print(f"  Using the provided Evo model as the shared base...")
            shared_evo_wrapper = evo_model
        else:
            raise ValueError("Either local_model_path+model_name or evo_model must be provided")

        # Now iterate through all folds using the shared base Evo model
        for fold_idx, fold_model_path in enumerate(fold_model_paths, 1):
            print(f"Loading fold {fold_idx} model (using the shared Evo base)...")

            # Load fold model checkpoint
            checkpoint = torch.load(fold_model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model_state_dict']

            # Detect label transform from first fold
            if fold_idx == 1 and checkpoint.get('label_transform') == 'log10':
                print(f"  Detected log10 label transform in checkpoint")

            # FIX #4: Create a fresh model wrapper using the SHARED base Evo
            # The LoRA modules will be loaded fresh for each fold, but the frozen Evo backbone is shared
            fold_model = finetune_module.EvoWithRegressionHead(
                shared_evo_wrapper,  # SHARED across all folds (FIX #4)
                hidden_dim=hidden_dim if hidden_dim else 512,
                dropout=0.2,
                use_lora=True,
                lora_r=lora_r if lora_r else 16,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                task_type=task_type
            )

            # FIX #4: Use strict=False to load only the matching keys (LoRA params + head)
            # The shared Evo backbone parameters are ignored since they're already loaded
            fold_model.load_state_dict(state_dict, strict=False)

            fold_model.eval()
            fold_model = fold_model.to(device)

            fold_preds = []
            with torch.no_grad():
                for batch_data in test_loader:
                    # FIX #1: Handle attention_mask from dataset
                    if len(batch_data) == 3:
                        input_ids, _, attention_mask = batch_data
                    else:
                        input_ids, _ = batch_data
                        attention_mask = None

                    input_ids = input_ids.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    # FIX #1: Pass attention_mask to model
                    preds = fold_model(input_ids, attention_mask)

                    if task_type == 'classification':
                        pred_classes = torch.argmax(preds, dim=1)
                        fold_preds.extend(pred_classes.cpu().numpy())
                    else:
                        fold_preds.extend(preds.cpu().numpy())

            all_fold_predictions.append(np.array(fold_preds))

            # FIX #4: Free memory - delete only the fold-specific model, keep the shared Evo
            del fold_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Ensemble predictions
        all_fold_predictions = np.array(all_fold_predictions)  # shape: (n_folds, n_samples)

        # Inverse transform if log10 was used
        if checkpoint.get('label_transform') == 'log10':
            all_fold_predictions = np.power(10.0, all_fold_predictions) - 1.0
            all_targets_original = np.power(10.0, all_targets_original) - 1.0
            print(f"  Inverse transformed predictions from log10 space")

        if task_type == 'regression':
            if ensemble_method == 'mean':
                ensemble_preds = np.mean(all_fold_predictions, axis=0)
                print("Ensembling with simple mean")
            elif ensemble_method == 'weighted':
                weights = np.array(fold_val_scores)
                weights = weights / np.sum(weights)
                ensemble_preds = np.average(all_fold_predictions, axis=0, weights=weights)
                print(f"Ensembling with weighted mean, weights: {weights}")
            elif ensemble_method == 'median':
                ensemble_preds = np.median(all_fold_predictions, axis=0)
                print("Ensembling with median")
            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")

            # Calculate metrics
            mse = mean_squared_error(all_targets_original, ensemble_preds)
            mae = mean_absolute_error(all_targets_original, ensemble_preds)
            r2 = r2_score(all_targets_original, ensemble_preds)
            pearson_corr, pearson_p = pearsonr(all_targets_original, ensemble_preds)
            spearman_corr, spearman_p = spearmanr(all_targets_original, ensemble_preds)

            test_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            }

            print(f"\nTest Set Results (ensemble prediction):")
            print(f"  MSE:       {mse:.6f}")
            print(f"  MAE:       {mae:.6f}")
            print(f"  R²:        {r2:.6f}")
            print(f"  Pearson:   {pearson_corr:.6f} (p={pearson_p:.2e})")
            print(f"  Spearman:  {spearman_corr:.6f} (p={spearman_p:.2e})")

        else:  # classification
            if ensemble_method == 'mean':
                ensemble_preds = np.round(np.mean(all_fold_predictions, axis=0)).astype(int)
                print("Ensembling with simple mean")
            elif ensemble_method == 'weighted':
                weights = np.array(fold_val_scores)
                weights = weights / np.sum(weights)
                ensemble_preds = np.round(np.average(all_fold_predictions, axis=0, weights=weights)).astype(int)
                print(f"Ensembling with weighted mean, weights: {weights}")
            elif ensemble_method == 'majority':
                # Majority voting
                from scipy.stats import mode
                ensemble_preds = mode(all_fold_predictions, axis=0).mode[0]
                print("Ensembling with majority voting")
            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")

            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            accuracy = accuracy_score(all_targets_original, ensemble_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets_original, ensemble_preds, average='binary')
            cm = confusion_matrix(all_targets_original, ensemble_preds)

            test_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }

            print(f"\nTest Set Results (ensemble prediction):")
            print(f"  Accuracy:  {accuracy:.6f}")
            print(f"  Precision: {precision:.6f}")
            print(f"  Recall:    {recall:.6f}")
            print(f"  F1:        {f1:.6f}")
            print(f"\nConfusion Matrix:")
            print(cm)

        # Set up variables for consistent output format
        ensemble_preds_original = ensemble_preds
        all_fold_predictions_original = all_fold_predictions

    else:
        # Single model evaluation
        model.eval()
        print(f"\n{'='*60}")
        print("Test Set Evaluation")
        print(f"{'='*60}")
        print(f"Test set size: {len(test_seqs)}")
        print(f"Task type: {task_type}")

        # Load label transform info from checkpoint
        label_transform = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            label_transform = checkpoint.get('label_transform', None)
            if label_transform:
                print(f"Detected label transform: {label_transform}")

        # Create test dataset
        test_dataset = PromoterDataset(test_seqs, test_labels, tokenizer, max_len=max_len,
                                       task_type=task_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for input_ids, labels, attention_mask in tqdm(test_loader, desc="Testing", leave=False):
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                predictions = model(input_ids)

                if task_type == 'classification':
                    pred_classes = torch.argmax(predictions, dim=1)
                    all_preds.extend(pred_classes.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                else:
                    all_preds.extend(predictions.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Inverse transform if log10 was used
        if label_transform == 'log10' and task_type == 'regression':
            all_preds = np.power(10.0, all_preds) - 1.0
            all_targets = np.power(10.0, all_targets) - 1.0
            print("Inverse transformed predictions and targets from log10 space")

        # Set up variables for consistent output format
        all_targets_original = all_targets
        ensemble_preds_original = all_preds
        all_fold_predictions_original = [all_preds]

        # Calculate metrics
        if task_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            accuracy = accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
            cm = confusion_matrix(all_targets, all_preds)

            test_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }

            print(f"\nTest Set Results:")
            print(f"  Accuracy:  {accuracy:.6f}")
            print(f"  Precision: {precision:.6f}")
            print(f"  Recall:    {recall:.6f}")
            print(f"  F1:        {f1:.6f}")
            print(f"\nConfusion Matrix:")
            print(cm)
        else:
            mse = mean_squared_error(all_targets, all_preds)
            mae = mean_absolute_error(all_targets, all_preds)
            r2 = r2_score(all_targets, all_preds)
            pearson_corr, pearson_p = pearsonr(all_targets, all_preds)
            spearman_corr, spearman_p = spearmanr(all_targets, all_preds)

            test_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            }

            print(f"\nTest Set Results:")
            print(f"  MSE:       {mse:.6f}")
            print(f"  MAE:       {mae:.6f}")
            print(f"  R²:        {r2:.6f}")
            print(f"  Pearson:   {pearson_corr:.6f} (p={pearson_p:.2e})")
            print(f"  Spearman:  {spearman_corr:.6f} (p={spearman_p:.2e})")

    # Save test results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'test_results.txt')

        with open(results_path, 'w') as f:
            f.write(f"Test Set Evaluation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Test set size: {len(test_seqs)}\n")
            f.write(f"Task type: {task_type}\n")

            if use_ensemble:
                f.write(f"Ensemble method: {ensemble_method}\n")
                f.write(f"Number of fold models: {len(fold_model_paths)}\n")

            f.write(f"\n")

            if task_type == 'classification':
                f.write(f"Accuracy:  {test_metrics['accuracy']:.6f}\n")
                f.write(f"Precision: {test_metrics['precision']:.6f}\n")
                f.write(f"Recall:    {test_metrics['recall']:.6f}\n")
                f.write(f"F1:        {test_metrics['f1']:.6f}\n\n")
                f.write(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}\n")
            else:
                f.write(f"MSE:       {test_metrics['mse']:.6f}\n")
                f.write(f"MAE:       {test_metrics['mae']:.6f}\n")
                f.write(f"R²:        {test_metrics['r2']:.6f}\n")
                f.write(f"Pearson:   {test_metrics['pearson_corr']:.6f} (p={test_metrics['pearson_p']:.2e})\n")
                f.write(f"Spearman:  {test_metrics['spearman_corr']:.6f} (p={test_metrics['spearman_p']:.2e})\n")

                if use_ensemble and ensemble_method == 'weighted':
                    f.write(f"\nPer-fold weights (based on validation Pearson r):\n")
                    for i, (w, r2) in enumerate(zip(weights / np.sum(weights) if 'weights' in locals() else fold_val_scores, fold_val_scores), 1):
                        f.write(f"  Fold {i}: weight={w:.4f}, val_Pearson_r={r2:.4f}\n")

        print(f"\nTest results saved to: {results_path}")

        # Save predictions to NPZ
        predictions_path = os.path.join(output_dir, 'test_predictions.npz')
        if task_type == 'regression':
            np.savez_compressed(
                predictions_path,
                sequences=np.array(test_seqs, dtype=object),
                true_labels=all_targets_original,
                predicted_labels=ensemble_preds_original
            )
        else:
            np.savez_compressed(
                predictions_path,
                sequences=np.array(test_seqs, dtype=object),
                true_labels=all_targets_original,
                predicted_labels=ensemble_preds_original if use_ensemble else all_preds
            )
        print(f"Predictions saved to: {predictions_path}")

        # Save detailed results to CSV (matching DeepBioParts format)
        csv_path = os.path.join(output_dir, 'test_results.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            if task_type == 'regression':
                header = ['Sequence', 'Ensemble_Prediction', 'Ground_Truth']
                for i in range(1, len(all_fold_predictions_original) + 1):
                    header.append(f'Fold_{i}_Prediction')
                writer.writerow(header)

                for idx in range(len(test_seqs)):
                    seq = test_seqs[idx]
                    true_label = all_targets_original[idx]
                    ensemble_pred = ensemble_preds_original[idx]

                    row_data = [seq, f'{ensemble_pred:.6f}', f'{true_label:.6f}']
                    for fold_preds in all_fold_predictions_original:
                        row_data.append(f'{fold_preds[idx]:.6f}')

                    writer.writerow(row_data)
            else:  # classification
                header = ['Sequence', 'Predicted_Class', 'Ground_Truth']
                writer.writerow(header)

                for idx in range(len(test_seqs)):
                    seq = test_seqs[idx]
                    true_label = int(all_targets_original[idx])
                    pred_label = int(ensemble_preds_original[idx] if use_ensemble else all_preds[idx])

                    writer.writerow([seq, pred_label, true_label])

        print(f"CSV results saved to: {csv_path}")

    print(f"{'='*60}\n")
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune Evo for biopart strength prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune promoter with default settings (5-fold cross-validation)
  python lora_finetune_evo.py --biopart promoter

  # Fine-tune RBS with default settings
  python lora_finetune_evo.py --biopart rbs

  # Fine-tune terminator with default settings
  python lora_finetune_evo.py --biopart terminator

  # Fine-tune with custom parameters
  python lora_finetune_evo.py --biopart promoter \\
      --model-name evo-1.5-8k-base \\
      --epochs 100 \\
      --lr 5e-5 \\
      --batch-size 64

  # Fine-tune with similarity-based data splitting (avoids similar sequences across splits)
  python lora_finetune_evo.py --biopart promoter \\
      --use-similarity-split \\
      --similarity-threshold 0.85 \\
      --kmer-size 3

  # Fine-tune with single train/val/test split (no K-fold CV)
  python lora_finetune_evo.py --biopart promoter \\
      --no-kfold-cv \\
      --test-size 0.2 \\
      --val-size 0.1

  # Fine-tune with custom number of folds
  python lora_finetune_evo.py --biopart promoter \\
      --n-folds 10 \\
      --ensemble-method weighted

  # Fine-tune without LoRA (use partial freezing)
  python lora_finetune_evo.py --biopart promoter \\
      --no-lora

  # Resume training from a checkpoint
  python lora_finetune_evo.py --biopart promoter \\
      --resume-from ./checkpoints/promoter/best_model.pth \\
      --epochs 100

  # Use custom data path and output directory (overrides --biopart defaults)
  python lora_finetune_evo.py --biopart promoter \\
      --data-path custom_data.csv \\
      --output-dir custom_output

Data Split Strategy:
  With K-fold cross-validation (default):
  - 85%% of data used for K-fold cross-validation (train+val)
  - 15%% held out as test set (evaluated after training)
  - Default: 5-fold CV with weighted ensemble prediction
  - Each fold: 68%% train / 17%% val (from the 85%%)
  - Final prediction: Weighted average of all 5 fold models
  - Output: test_results.csv with ensemble and individual fold predictions

  With --no-kfold-cv (single split mode):
  - 70%% training set
  - 15%% validation set (for early stopping)
  - 15%% test set (held out until final evaluation)

  With --use-similarity-split:
  - Uses cluster-based splitting to avoid similar sequences across splits
  - Ensures more reliable test set evaluation by preventing data leakage
        """
    )

    parser.add_argument('--biopart', type=str, choices=['promoter', 'rbs', 'terminator'],
                        help='Type of biopart to fine-tune (promoter, rbs, or terminator). '
                             'This will automatically set appropriate --data-path and --output-dir.')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to biopart dataset CSV (must have "sequence" and "activity" columns). '
                             'If not specified, will be automatically set based on --biopart.')
    parser.add_argument('--model-name', type=str, default='evo-1.5-8k-base',
                        choices=['evo-1.5-8k-base', 'evo-1-131k-base'],
                        help='Evo model name (default: evo-1.5-8k-base)')
    parser.add_argument('--local-model-path', type=str, default=None,
                        help='Path to local model directory (default: ./models/<model-name>)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints. '
                             'If not specified, will be automatically set based on --biopart.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32, increased for more stable gradients)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5, reduced for stability)')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='Hidden dimension of regression head (default: 1024, increased for better capacity)')
    parser.add_argument('--max-len', type=int, default=None,
                        help='Maximum sequence length (default: auto-detect from data using 95th percentile)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1, reduced for better learning)')
    parser.add_argument('--use-lora', action='store_true', default=True,
                        help='Use LoRA for efficient fine-tuning (default: True)')
    parser.add_argument('--no-lora', dest='use_lora', action='store_false',
                        help='Do not use LoRA, use partial freezing instead')
    parser.add_argument('--lora-r', type=int, default=64,
                        help='LoRA rank (default: 64, increased for better representation learning)')
    parser.add_argument('--lora-alpha', type=int, default=128,
                        help='LoRA alpha scaling factor (default: 128, typically 2x rank)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout rate (default: 0.1, reduced for better learning)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--use-similarity-split', action='store_true', default=True,
                        help='Use similarity-based cluster splitting to avoid similar sequences across train/val/test sets')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                        help='Maximum allowed similarity between sequences in different splits (0-1, default: 0.7). '
                             'Lower values = more strict splitting (reduces data leakage)')
    parser.add_argument('--kmer-size', type=int, default=3,
                        help='k-mer size for similarity computation (default: 3)')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set ratio (default: 0.15, i.e., 15%% for test set)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Validation set ratio from remaining data (default: 0.2, i.e., 20%% of remaining for val). '
                             'Final split: 70%% train / 15%% val / 15%% test with default settings.')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Number of epochs to wait for improvement before early stopping (default: 15)')
    parser.add_argument('--log-label', action='store_true',
                        help='Apply log10(label+1) transform to labels (for terminator)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume training from (default: None)')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing to reduce memory usage (trades computation for memory, default: True)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                        help='Number of steps to accumulate gradients before updating (default: 2, reduced with larger batch)')
    parser.add_argument('--use-kfold-cv', action='store_true', default=False,
                        help='Use K-fold cross-validation for training (default: False)')
    parser.add_argument('--no-kfold-cv', dest='use_kfold_cv', action='store_false',
                        help='Do not use K-fold cross-validation, use single train/val/test split instead (default)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--ensemble-method', type=str, default='weighted', choices=['mean', 'weighted', 'median'],
                        help='Ensemble method for combining fold model predictions (default: weighted)')

    args = parser.parse_args()

    # Validate that --biopart is provided
    if args.biopart is None:
        parser.error("--biopart is required. Please specify one of: promoter, rbs, terminator")

    # Set default data-path and output-dir based on biopart if not explicitly provided
    current_date = datetime.now().strftime("%Y%m%d")

    # Define biopart-specific configurations
    biopart_configs = {
        'promoter': {'output_dir': f'checkpoints/promoter_{current_date}', 'task_type': 'regression'},
        'rbs': {'output_dir': f'checkpoints/rbs_{current_date}', 'task_type': 'regression'},
        'terminator': {'output_dir': f'checkpoints/terminator_{current_date}', 'task_type': 'regression'}
    }

    # The training CSV must always be provided explicitly
    if args.data_path is None:
        raise SystemExit("--data_path is required (no default dataset path is bundled)")

    if args.output_dir is None:
        args.output_dir = biopart_configs[args.biopart]['output_dir']
        print(f"Using default output directory for {args.biopart}: {args.output_dir}")

    # Determine task type based on biopart
    task_type = biopart_configs[args.biopart]['task_type']
    print(f"Task type: {task_type}")

    # Run fine-tuning
    fold_model_paths, fold_val_scores, test_seqs, test_labels, max_len = finetune_evo_promoter_strength(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=args.device,
        local_model_path=args.local_model_path,
        use_similarity_split=args.use_similarity_split,
        similarity_threshold=args.similarity_threshold,
        kmer_size=args.kmer_size,
        resume_from=args.resume_from,
        max_len=args.max_len,
        early_stopping_patience=args.early_stopping_patience,
        task_type=task_type,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        test_size=args.test_size,
        val_size=args.val_size,
        n_folds=args.n_folds,
        use_kfold_cv=args.use_kfold_cv,
        use_log_label=args.log_label
    )

    # Run test set evaluation if test set is available
    if test_seqs is not None and len(test_seqs) > 0:
        print("\n" + "="*60)
        print("Training and validation completed.")
        print("Running test set evaluation with ensemble prediction...")
        print("="*60)

        # Clean up GPU memory before loading test model
        print("Cleaning up GPU memory...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Print memory status after cleanup
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(args.device) / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(args.device) / 1024**3:.2f} GB")

        # Load tokenizer for test evaluation
        evo_wrapper = load_evo_from_local(args.local_model_path, args.model_name, args.device)
        tokenizer = evo_wrapper.tokenizer

        # Run test evaluation with ensemble
        test_metrics = test_model(
            model=None,  # None for ensemble prediction
            test_seqs=test_seqs,
            test_labels=test_labels,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            task_type=task_type,
            output_dir=os.path.join(args.output_dir, 'test_results'),
            max_len=max_len,
            fold_model_paths=fold_model_paths,
            fold_val_scores=fold_val_scores,
            ensemble_method=args.ensemble_method,
            evo_model=evo_wrapper,  # Pass Evo model wrapper (for fallback)
            local_model_path=args.local_model_path,  # Pass for fresh loading
            model_name=args.model_name  # Pass for fresh loading
        )

        print("\n" + "="*60)
        print("Test set evaluation completed!")
        print(f"Results saved to: {os.path.join(args.output_dir, 'test_results')}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Training and validation completed.")
        print("No test set available for evaluation.")
        print("="*60)
