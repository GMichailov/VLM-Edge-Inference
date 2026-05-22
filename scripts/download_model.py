"""Download a model from Hugging Face and save it under models/<name>."""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("model_id", help="Full HF model ID, e.g. 'org/model-name'")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(message)s")

    model_name = args.model_id.split("/")[-1]
    local_dir = Path(__file__).resolve().parent.parent / "models" / model_name

    logger.info("Downloading %s -> %s", args.model_id, local_dir)
    snapshot_download(repo_id=args.model_id, local_dir=str(local_dir))
    logger.info("Done")


if __name__ == "__main__":
    main()
