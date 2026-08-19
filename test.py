#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="FBDNet inference")
    parser.add_argument("--config", default="configs/fbdnet_m5.py")
    parser.add_argument("--checkpoint", default="checkpoints/fbdnet.pth")
    parser.add_argument("--image", default="demo/sample.png")
    parser.add_argument("--out-dir", default="outputs/fbdnet")
    args = parser.parse_args()

    print("Config:", Path(args.config))
    print("Checkpoint:", Path(args.checkpoint))
    print("Image:", Path(args.image))
    print("Output dir:", Path(args.out_dir))
    print("Inference pipeline: semantic, boundary, distance, adaptive decoding.")


if __name__ == "__main__":
    main()
