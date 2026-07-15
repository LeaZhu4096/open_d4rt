"""Audit whether an OpenD4RT checkpoint matches a model.yaml configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.checkpoint import load_checkpoint
from src.model.builder import build_model


def unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--max-lines", type=int, default=40)
    args = parser.parse_args()

    model_config = Path(args.model_config)
    ckpt_path = Path(args.ckpt_path)
    if not model_config.exists():
        raise FileNotFoundError(model_config)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with model_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"])
    model_state = model.state_dict()
    ckpt_state = unwrap_state_dict(torch.load(Path(ckpt_path), map_location="cpu", mmap=True))
    if not ckpt_state:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")

    matched = []
    shape_mismatch = []
    missing = []
    for key, tensor in model_state.items():
        src = ckpt_state.get(key)
        if src is None:
            missing.append(key)
        elif tuple(src.shape) == tuple(tensor.shape):
            matched.append(key)
        else:
            shape_mismatch.append((key, tuple(src.shape), tuple(tensor.shape)))

    unexpected = [key for key in ckpt_state.keys() if key not in model_state]
    print(f"model_tensors={len(model_state)} checkpoint_tensors={len(ckpt_state)}")
    print(f"matched={len(matched)} missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}")
    print(f"matched_params={sum(model_state[k].numel() for k in matched):,}")
    print(f"model_params={sum(v.numel() for v in model_state.values()):,}")

    if shape_mismatch:
        print("\nshape_mismatch examples:")
        for key, src_shape, dst_shape in shape_mismatch[: args.max_lines]:
            print(f"  {key}: ckpt={src_shape} model={dst_shape}")
    if missing:
        print("\nmissing examples:")
        for key in missing[: args.max_lines]:
            print(f"  {key}")
    if unexpected:
        print("\nunexpected examples:")
        for key in unexpected[: args.max_lines]:
            print(f"  {key}")


if __name__ == "__main__":
    main()