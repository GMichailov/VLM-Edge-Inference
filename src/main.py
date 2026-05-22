"""VLM Edge Inference — main entry point."""

import argparse
import logging
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Edge Inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=240)
    parser.add_argument("--use_chat_template", type=bool, default=True)
    parser.add_argument("--max_ram_usage", type=float, default=None, help="Max RAM usage in GB")
    args = parser.parse_args()

    if not args.prompt:
        parser.error("--prompt must not be empty")

    model_dir = MODELS_DIR / args.model
    if not model_dir.exists():
        parser.error(f"Model not found: {model_dir}")

    trace_path = LOGS_DIR / f"{args.model}_trace.json"
    save_path = LOGS_DIR / f"{args.model}.saveTensors"
    if not trace_path.exists():
        parser.error(f"Trace not found: {trace_path} (run generate_architecture_report.py first)")
    if not save_path.exists():
        parser.error(f"Save plan not found: {save_path} (run create_inference_plan.py first)")

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

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model_dir = MODELS_DIR / args.model

    from inference.engine import InferenceEngine
    engine = InferenceEngine(
        model_dir, args.prompt, args.image_height, args.image_width, args.use_chat_template
    )
    engine.inference()


if __name__ == "__main__":
    main()
