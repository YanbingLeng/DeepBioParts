"""Evo predictor architecture and pretrained-checkpoint loading for inference.

The module preserves the released LoRA and pooling-head state_dict keys.
Evo runtime dependencies are imported only when loading/tokenizing a model.
"""

import os
import torch
import torch.nn as nn
import yaml


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
    


def apply_lora_to_linear(linear_module, rank=16, alpha=32, dropout=0.1):
    """
    Replace a Linear module with a LoRA-enabled version.

    LoRA parameters use the same dtype as the original module (typically bfloat16 for Evo).
    """
    # Use the same dtype as the original module for LoRA parameters
    # Preserve the base model dtype when restoring the adapter
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
        return output.squeeze(-1)


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
                 evo_adaptation=None, pooling_mode='attention'):
        """
        Args:
            task_type: 'regression' for promoter/rbs strength prediction,
                      'classification' for terminator binary classification
            evo_adaptation: How the Evo backbone is adapted (ablation). When None, it is
                inferred from ``use_lora`` for backward compatibility
                (use_lora=True -> 'lora', use_lora=False -> 'partial_ft').
                - 'lora'       : LoRA injection
                - 'head_only'  : load the pooling/output head without a LoRA adapter
                - 'partial_ft' : load backbone weights adapted without LoRA
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
        """Predict activity or class logits from Evo hidden states."""
        x = self.evo.embedding_layer(input_ids)
        for block in self.evo.blocks:
            x, _ = block(x)
        x = self.evo.norm(x)
        return self.head(x.float(), attention_mask)



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
