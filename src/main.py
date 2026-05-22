"""VLM Edge Inference — main entry point."""

import argparse
import logging
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser(description="VLM Edge Inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=240)
    parser.add_argument("--use_chat_template", type=bool, default=True)
    parser.add_argument("--max_ram_usage", type=float, default=None, help="Max RAM usage in GB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.prompt:
        parser.error("--prompt must not be empty")

    model_dir = MODELS_DIR / args.model
    if not model_dir.exists():
        parser.error(f"Model not found: {model_dir}")

    if args.image_height > 480:
        logger.warning("Clamping image_height to 480")
        args.image_height = 480
    if args.image_width > 480:
        logger.warning("Clamping image_width to 480")
        args.image_width = 480

    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    cap_gb = total_gb * 0.8
    if args.max_ram_usage is None:
        args.max_ram_usage = cap_gb
    elif args.max_ram_usage > cap_gb:
        logger.warning("Requested %.1f GB exceeds 80%% of system RAM (%.1f GB), capping", args.max_ram_usage, cap_gb)
        args.max_ram_usage = cap_gb


if __name__ == "__main__":
    main()
