"""Dynamic module loading and execution for memory-efficient inference."""

import copy
import json
import logging

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

logger = logging.getLogger(__name__)


class ModuleLoader:
    """Builds a weightless model skeleton once and extracts modules from it."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._index = self._load_index()
        self._skeleton = self._build_skeleton()
        self._modules_cache = dict(self._skeleton.named_modules())

    def _load_index(self):
        index_path = self.model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        logger.info("Loaded safetensors index: %d weights", len(index["weight_map"]))
        return index

    def _build_skeleton(self):
        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            import json as _json
            config_dict = _json.load(f)

        from transformers import AutoConfig
        config = AutoConfig.for_model(**config_dict)

        logger.info("Building model skeleton from config")
        skeleton = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
        skeleton.eval()
        logger.info("Skeleton built: %d sub-modules", len(list(skeleton.named_modules())))
        return skeleton

    def get_module(self, module_name):
        module = self._modules_cache.get(module_name)
        if module is None:
            raise KeyError(f"Module '{module_name}' not found in skeleton")
        return copy.deepcopy(module)

    def load_weights(self, module, trace_entry):
        weight_map = self._index["weight_map"]
        module_name = trace_entry["module_name"]
        param_info = {p["name"]: p["dtype"] for p in trace_entry["parameters"]}

        # Group params by shard file
        shard_params = {}
        for pname, expected_dtype in param_info.items():
            full_key = f"{module_name}.{pname}"
            shard_file = weight_map.get(full_key)
            if shard_file is None:
                raise KeyError(f"Weight '{full_key}' not found in index")
            shard_params.setdefault(shard_file, []).append((pname, full_key, expected_dtype))

        state_dict = {}
        for shard_file, entries in shard_params.items():
            shard_path = self.model_dir / shard_file
            with safe_open(shard_path, framework="pt") as f:
                for pname, full_key, expected_dtype in entries:
                    tensor = f.get_tensor(full_key)
                    dtype = getattr(torch, expected_dtype.split(".")[-1], None)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    state_dict[pname] = tensor

        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys for %s: %s", module_name, missing)
        if unexpected:
            logger.warning("Unexpected keys for %s: %s", module_name, unexpected)

        logger.info("Loaded %d weights for %s", len(state_dict), module_name)

    def free_skeleton(self):
        del self._skeleton
        del self._modules_cache
        self._skeleton = None
        self._modules_cache = None
        logger.info("Freed model skeleton")


class ModuleWrapper:
    """Wraps a single loaded module for execution."""

    def __init__(self, loader, trace_entry):
        self.loader = loader
        self.trace_entry = trace_entry
        self.module = None

    def load(self):
        self.module = self.loader.get_module(self.trace_entry["module_name"])
        self.loader.load_weights(self.module, self.trace_entry)

    def forward(self, **input_tensors):
        if self.module is None:
            raise RuntimeError("Module not loaded — call load() first")
        return self.module(**input_tensors)

    def unload(self):
        del self.module
        self.module = None
