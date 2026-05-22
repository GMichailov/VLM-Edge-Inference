"""Trace every module's forward pass: shapes, dtypes, parameters, execution order."""

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


def _extract_tensors(obj):
    """Recursively find all tensors in a nested structure."""
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


def _make_hook(name, log, counter):
    def hook(module, args, kwargs, output):
        counter[0] += 1

        input_tensors = _extract_tensors(args) + _extract_tensors(kwargs)
        output_tensors = _extract_tensors(output)

        log.append({
            "execution_order": counter[0],
            "module_name": name,
            "module_class": module.__class__.__name__,
            "inputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in input_tensors],
            "outputs": [{"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)} for t in output_tensors],
            "parameters": [{"name": n, "shape": list(p.shape)} for n, p in module.named_parameters()],
        })

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

    log = []
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
