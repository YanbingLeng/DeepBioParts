# Vendored from the Apache-2.0 licensed Evo model package (evo-model 0.4,
# https://github.com/evo-design/evo). MODIFIED for DeepBioParts: added
# offline loading of local checkpoints and robust cached-generation
# fallback for LoRA-adapted models. See THIRD_PARTY_NOTICES.md.
import json
import pkgutil
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM
import yaml

from stripedhyena.utils import dotdict
from stripedhyena.model import StripedHyena
from stripedhyena.tokenizer import CharLevelTokenizer


MODEL_NAMES = [
    'evo-1.5-8k-base',
    'evo-1-8k-base',
    'evo-1-131k-base',
    'evo-1-8k-crispr',
    'evo-1-8k-transposon',
]

class Evo:
    def __init__(
        self,
        model_name: str = MODEL_NAMES[1],
        device: str = None,
        local_model_path: str = None,
    ):
        """
        Loads an Evo model checkpoint given a model name.
        If the checkpoint does not exist, we automatically download it from HuggingFace.
        """
        self.device = device

        # Check model name.

        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Assign config path.

        if model_name == 'evo-1-8k-base' or \
           model_name == 'evo-1-8k-crispr' or \
           model_name == 'evo-1-8k-transposon' or \
           model_name == 'evo-1.5-8k-base':
            config_path = 'configs/evo-1-8k-base_inference.yml'
        elif model_name == 'evo-1-131k-base':
            config_path = 'configs/evo-1-131k-base_inference.yml'
        else:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Load model.

        if local_model_path is None:
            self.model = load_checkpoint(
                model_name=model_name,
                config_path=config_path,
                device=self.device
            )
        else:
            self.model = load_local_checkpoint(
                local_model_path=local_model_path,
                config_path=config_path,
                device=self.device,
            )

        # Load tokenizer.

        self.tokenizer = CharLevelTokenizer(512)

        
HF_MODEL_NAME_MAP = {
    'evo-1.5-8k-base': 'evo-design/evo-1.5-8k-base',
    'evo-1-8k-base': 'togethercomputer/evo-1-8k-base',
    'evo-1-131k-base': 'togethercomputer/evo-1-131k-base',
    'evo-1-8k-crispr': 'LongSafari/evo-1-8k-crispr',
    'evo-1-8k-transposon': 'LongSafari/evo-1-8k-transposon',
}


def _load_local_state_dict(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_local_checkpoint(
    local_model_path: str,
    config_path: str = 'evo/configs/evo-1-8k-base_inference.yml',
    device: str = None,
):
    """Load a sharded Evo checkpoint exclusively from a local directory."""
    checkpoint_dir = Path(local_model_path).expanduser().resolve()
    index_path = checkpoint_dir / 'pytorch_model.bin.index.json'
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f'Local Evo model directory not found: {checkpoint_dir}')
    if not index_path.is_file():
        raise FileNotFoundError(f'Local Evo checkpoint index not found: {index_path}')

    with index_path.open('r', encoding='utf-8') as handle:
        weight_map = json.load(handle)['weight_map']
    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (checkpoint_dir / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f'Missing local Evo checkpoint shards in {checkpoint_dir}: {", ".join(missing_shards)}'
        )

    config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
    global_config = dotdict(config, Loader=yaml.FullLoader)
    model = StripedHyena(global_config)
    model.to_bfloat16_except_poles_residues()
    model_keys = set(model.state_dict())
    loaded_keys = set()

    print(f'Loading Evo checkpoint only from local directory: {checkpoint_dir}')
    for shard_name in shard_names:
        shard_path = checkpoint_dir / shard_name
        shard_state = _load_local_state_dict(shard_path)
        if not all(key.startswith('backbone.') for key in shard_state):
            raise RuntimeError(f'Unexpected key format in local Evo shard: {shard_path}')
        local_state = {key.removeprefix('backbone.'): value for key, value in shard_state.items()}
        unexpected = set(local_state) - model_keys
        if unexpected:
            preview = ', '.join(sorted(unexpected)[:5])
            raise RuntimeError(f'Unexpected keys in local Evo shard {shard_name}: {preview}')
        model.load_state_dict(local_state, strict=False)
        loaded_keys.update(local_state)
        del shard_state, local_state

    missing_keys = model_keys - loaded_keys
    if missing_keys:
        preview = ', '.join(sorted(missing_keys)[:5])
        raise RuntimeError(f'Local Evo checkpoint is incomplete; missing keys include: {preview}')
    if device is not None:
        model = model.to(device)
    return model

def load_checkpoint(
    model_name: str = MODEL_NAMES[1],
    config_path: str = 'evo/configs/evo-1-131k-base_inference.yml',
    device: str = None,
    *args, **kwargs
):
    """
    Load checkpoint from HuggingFace and place it into SH model.
    """

    # Map model name to HuggingFace model name.

    hf_model_name = HF_MODEL_NAME_MAP[model_name]

    # Load model config.

    model_config = AutoConfig.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        revision='1.1_fix' if re.match(r'evo-1-.*-base', model_name) else 'main',
    )
    model_config.use_cache = True

    # Load model.

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        config=model_config,
        trust_remote_code=True,
        revision='1.1_fix' if re.match(r'evo-1-.*-base', model_name) else 'main',
    )

    # Load model state dict & cleanup.

    state_dict = model.backbone.state_dict()
    del model
    del model_config

    # Load SH config.

    config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
    global_config = dotdict(config, Loader=yaml.FullLoader)

    # Load SH Model.

    model = StripedHyena(global_config)
    model.load_state_dict(state_dict, strict=True)
    model.to_bfloat16_except_poles_residues()
    if device is not None:
        model = model.to(device)

    return model
