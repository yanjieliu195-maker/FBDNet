#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("fbdnet_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="FBDNet training")
    parser.add_argument("--config", default="configs/fbdnet_m5.py")
    parser.add_argument("--data-root", default="data/fbdnet")
    parser.add_argument("--work-dir", default="work_dirs/fbdnet_m5")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))

    print("Config:", cfg_path)
    print("Model:", cfg.model["name"])
    print("Data root:", Path(args.data_root))
    print("Work dir:", Path(args.work_dir))


if __name__ == "__main__":
    main()
