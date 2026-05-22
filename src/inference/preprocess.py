"""Preprocess images and prompt into model-ready inputs."""

import logging
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "images"
UNPROCESSED_DIR = IMAGES_DIR / "unprocessed"


def preprocess(model_dir, prompt, height, width, use_chat_template=True):
    """Load images, pair each with the prompt, run AutoProcessor on the batch."""
    if not UNPROCESSED_DIR.exists():
        logger.error("No unprocessed images directory: %s", UNPROCESSED_DIR)
        return None
    images = []
    for path in sorted(UNPROCESSED_DIR.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            continue
        img = Image.open(path).convert("RGB").resize((width, height))
        images.append(img)
        logger.info("Loaded %s (%dx%d)", path.name, width, height)
    if not images:
        logger.error("No images found in %s", UNPROCESSED_DIR)
        return None
    processor = AutoProcessor.from_pretrained(model_dir)
    texts = []
    for _ in images:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        texts.append(messages)
    if use_chat_template:
        texts = [processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in texts]
    inputs = processor(text=texts, images=images, return_tensors="pt")
    del processor
    logger.info("Preprocessed %d images into batch inputs", len(images))
    return inputs


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    project_root = Path(__file__).resolve().parent.parent.parent
    model_dir = project_root / "models" / (sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-0.8B")

    inputs = preprocess(model_dir, "Describe this image.", 240, 240)
    if inputs is not None:
        for key, val in inputs.items():
            if hasattr(val, "shape"):
                print(f"  {key}: shape={list(val.shape)}, dtype={val.dtype}")
            else:
                print(f"  {key}: {type(val).__name__} = {val}")
