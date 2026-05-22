"""Memory-efficient embedding lookup using safetensors slices."""

import json
import logging

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def embed(model_dir, module_name, input_ids):
    """Look up embeddings row-by-row from safetensors without loading the full weight matrix.

    Args:
        model_dir: Path to the model directory.
        module_name: e.g. "model.language_model.embed_tokens"
        input_ids: tensor of token IDs

    Returns:
        inputs_embeds tensor of shape matching the embedding table rows for each ID.
    """
    index_path = model_dir / "model.safetensors.index.json"
    weight_key = f"{module_name}.weight"

    with open(index_path) as f:
        index = json.load(f)

    shard_file = index["weight_map"].get(weight_key)
    if shard_file is None:
        raise KeyError(f"{weight_key} not found in safetensors index")

    shard_path = model_dir / shard_file

    unique_ids = torch.unique(input_ids).tolist()
    rows = {}
    with safe_open(shard_path, framework="pt") as f:
        emb = f.get_slice(weight_key)
        for token_id in unique_ids:
            rows[token_id] = emb[token_id:token_id + 1]

    inputs_embeds = torch.cat([rows[tid] for tid in input_ids.flatten().tolist()], dim=0)
    inputs_embeds = inputs_embeds.view(*input_ids.shape, -1)

    logger.info("Embedded %d tokens (%d unique) from %s", input_ids.numel(), len(unique_ids), shard_file)
    return inputs_embeds
