"""Trace every module's forward pass: shapes, dtypes, parameters, execution order."""

import argparse
import inspect
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

_COMPOSITE_MODULES = frozenset({
    "Qwen3_5DecoderLayer", "Qwen3_5VisionBlock", "Qwen3_5VisionModel",
    "Qwen3_5TextModel", "Qwen3_5Model", "Qwen3_5ForConditionalGeneration",
})


def _extract_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if obj is None:
        return []
    if isinstance(obj, (tuple, list)):
        out = []
        for x in obj:
            out.extend(_extract_tensors(x))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_extract_tensors(v))
        return out
    return []


def _add(overhead, dtype_str, count):
    if count > 0:
        overhead[dtype_str] = overhead.get(dtype_str, 0) + count


def _add_attention_intermediates(overhead, module, input_tensors):
    # Text standard attention: GQA + attn_output_gate + SDPA on CPU
    num_heads = getattr(module, "num_heads", None)
    head_dim = getattr(module, "head_dim", None)
    num_kv_heads = getattr(module, "num_key_value_heads", num_heads)
    if num_heads is None or head_dim is None:
        return

    B, S = input_tensors[0].shape[0], input_tensors[0].shape[1]
    dtype_str = str(input_tensors[0].dtype)
    fp32 = "torch.float32"

    # Projection outputs
    _add(overhead, dtype_str, B * S * 2 * num_heads * head_dim)      # q_proj (fused Q+gate)
    _add(overhead, dtype_str, B * S * num_kv_heads * head_dim)       # k_proj
    _add(overhead, dtype_str, B * S * num_kv_heads * head_dim)       # v_proj

    # GQA expansion (repeat_kv materialises new tensors)
    _add(overhead, dtype_str, B * num_heads * S * head_dim)          # K expanded
    _add(overhead, dtype_str, B * num_heads * S * head_dim)          # V expanded

    # SDPA float32 intermediates (CPU math backend upcasts bf16)
    _add(overhead, fp32, 3 * B * num_heads * S * head_dim)           # Q, K, V upcast
    _add(overhead, fp32, B * num_heads * S * S)                      # attention scores
    _add(overhead, fp32, B * num_heads * S * S)                      # softmax
    _add(overhead, fp32, B * num_heads * S * head_dim)               # softmax @ V
    _add(overhead, dtype_str, B * num_heads * S * head_dim)          # cast back to bf16

    # Gate: sigmoid output and element-wise multiply
    _add(overhead, dtype_str, B * S * num_heads * head_dim)          # sigmoid(gate)
    _add(overhead, dtype_str, B * num_heads * S * head_dim)          # o_proj input (reshaped attn_output)


def _add_vision_attention_intermediates(overhead, module, input_tensors):
    # Vision attention: no batch dim in inputs, no GQA, no gate, SDPA on CPU
    num_heads = getattr(module, "num_heads", None)
    head_dim = getattr(module, "head_dim", None)
    if num_heads is None or head_dim is None:
        return

    S = input_tensors[0].shape[0]
    H_vis = input_tensors[0].shape[1]
    B = 1  # added internally
    dtype_str = str(input_tensors[0].dtype)
    fp32 = "torch.float32"

    # Fused QKV projection output
    _add(overhead, dtype_str, S * 3 * H_vis)

    # SDPA float32 intermediates
    _add(overhead, fp32, 3 * B * num_heads * S * head_dim)
    _add(overhead, fp32, B * num_heads * S * S)
    _add(overhead, fp32, B * num_heads * S * S)
    _add(overhead, fp32, B * num_heads * S * head_dim)
    _add(overhead, dtype_str, B * num_heads * S * head_dim)          # cast back


def _add_deltanet_intermediates(overhead, module, input_tensors):
    # Linear attention: no SDPA, all intermediates stay in native dtype
    B, S = input_tensors[0].shape[0], input_tensors[0].shape[1]
    dtype_str = str(input_tensors[0].dtype)

    param_dims = {}
    for name, p in module.named_parameters():
        if name in ("in_proj_qkv.weight", "in_proj_z.weight",
                     "in_proj_b.weight", "in_proj_a.weight"):
            param_dims[name] = p.shape[0]

    if "in_proj_qkv.weight" in param_dims:
        qkv_dim = param_dims["in_proj_qkv.weight"]
        _add(overhead, dtype_str, B * S * qkv_dim)                  # qkv projection
        _add(overhead, dtype_str, B * S * qkv_dim)                  # conv1d output
    if "in_proj_z.weight" in param_dims:
        _add(overhead, dtype_str, B * S * param_dims["in_proj_z.weight"])
    if "in_proj_b.weight" in param_dims:
        _add(overhead, dtype_str, B * S * param_dims["in_proj_b.weight"])
    if "in_proj_a.weight" in param_dims:
        _add(overhead, dtype_str, B * S * param_dims["in_proj_a.weight"])

    # Gated norm output / out_proj input (z_dim if present)
    if "in_proj_z.weight" in param_dims:
        _add(overhead, dtype_str, B * S * param_dims["in_proj_z.weight"])


def _add_mlp_intermediates(overhead, module, input_tensors):
    # SwiGLU: gate_proj -> SiLU -> * up_proj -> down_proj
    B, S = input_tensors[0].shape[0], input_tensors[0].shape[1]
    dtype_str = str(input_tensors[0].dtype)

    intermediate_size = None
    for name, p in module.named_parameters():
        if name == "gate_proj.weight":
            intermediate_size = p.shape[0]
            break
    if intermediate_size is None:
        return

    _add(overhead, dtype_str, B * S * intermediate_size)            # gate_proj(x)
    _add(overhead, dtype_str, B * S * intermediate_size)            # SiLU(gate)
    _add(overhead, dtype_str, B * S * intermediate_size)            # up_proj(x)
    _add(overhead, dtype_str, B * S * intermediate_size)            # gate * up


def _add_vision_mlp_intermediates(overhead, module, input_tensors):
    # GELU MLP: linear_fc1 -> GELU -> linear_fc2 (no batch dim)
    S = input_tensors[0].shape[0]
    dtype_str = str(input_tensors[0].dtype)

    intermediate_size = None
    for name, p in module.named_parameters():
        if name == "linear_fc1.weight":
            intermediate_size = p.shape[0]
            break
    if intermediate_size is None:
        return

    _add(overhead, dtype_str, S * intermediate_size)                # linear_fc1(x)
    _add(overhead, dtype_str, S * intermediate_size)                # GELU(fc1)


def _add_intermediates(overhead, module, input_tensors):
    cls_name = module.__class__.__name__
    try:
        if cls_name == "Qwen3_5Attention":
            _add_attention_intermediates(overhead, module, input_tensors)
        elif cls_name == "Qwen3_5VisionAttention":
            _add_vision_attention_intermediates(overhead, module, input_tensors)
        elif cls_name == "Qwen3_5GatedDeltaNet":
            _add_deltanet_intermediates(overhead, module, input_tensors)
        elif cls_name == "Qwen3_5MLP":
            _add_mlp_intermediates(overhead, module, input_tensors)
        elif cls_name == "Qwen3_5VisionMLP":
            _add_vision_mlp_intermediates(overhead, module, input_tensors)
    except Exception as e:
        logger.warning("Failed to compute intermediates for %s (%s): %s",
                       module, cls_name, e)


def _compute_overhead(module, input_tensors, output_tensors):
    if module.__class__.__name__ in _COMPOSITE_MODULES:
        return {}

    overhead = {}

    for t in input_tensors:
        _add(overhead, str(t.dtype), t.numel())

    for p in module.parameters():
        _add(overhead, str(p.dtype), p.numel())

    for t in output_tensors:
        _add(overhead, str(t.dtype), t.numel())

    _add_intermediates(overhead, module, input_tensors)

    return overhead


def _get_forward_kwargs(module):
    """Extract kwarg names and defaults from a module's forward signature."""
    try:
        sig = inspect.signature(module.forward)
    except (ValueError, TypeError):
        return {}
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD):
            continue
        if p.name == "self":
            continue
        kwargs[p.name] = None if p.default is p.empty else p.default
    return kwargs


def _make_hook(name, log, counter):
    def hook(module, args, kwargs, output):
        counter[0] += 1

        input_tensors = _extract_tensors(args) + _extract_tensors(kwargs)
        output_tensors = _extract_tensors(output)

        entry = {
            "module_name": name,
            "module_class": module.__class__.__name__,
            "forward_args": _get_forward_kwargs(module),
            "inputs": [{"id": id(t), "shape": list(t.shape), "dtype": str(t.dtype)} for t in input_tensors],
            "outputs": [{"id": id(t), "shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)} for t in output_tensors],
            "parameters": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in module.named_parameters()],
        }

        overhead = _compute_overhead(module, input_tensors, output_tensors)
        if overhead:
            entry["overhead"] = overhead

        log[counter[0]] = entry

    return hook


def main():
    parser = argparse.ArgumentParser(description="Trace module forward passes for a VLM")
    parser.add_argument("model_name", help="Model folder name under models/, e.g. Qwen3.5-0.8B")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(message)s")

    model_dir = Path(__file__).resolve().parent.parent / "models" / args.model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    logger.info("Loading processor from %s", model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    logger.info("Loading model from %s", model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16
    )
    model.eval()

    log = {}
    counter = [0]
    hooks = []
    for name, module in model.named_modules():
        h = module.register_forward_hook(_make_hook(name, log, counter), with_kwargs=True)
        hooks.append(h)
    logger.info("Registered %d forward hooks", len(hooks))

    fake_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": fake_image},
            {"type": "text", "text": "Describe this image."},
        ]}
    ]

    logger.info("Preparing fake inputs")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[fake_image], return_tensors="pt")

    logger.info("Running forward pass")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    output_path = Path(__file__).resolve().parent.parent / "logs" / f"{args.model_name}_trace.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)

    logger.info("Traced %d module calls -> %s", len(log), output_path)


if __name__ == "__main__":
    main()
