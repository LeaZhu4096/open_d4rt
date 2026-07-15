"""Validate that OpenD4RT full forward equals encoder + decoder split forward."""

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

OUTPUT_NAMES = ["xyz_3d", "uv_2d", "visibility", "displacement", "normal", "confidence"]


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
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    with Path(args.model_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"]).eval()
    state_dict = unwrap_state_dict(torch.load(Path(args.ckpt_path), map_location="cpu", mmap=True))
    result = model.load_state_dict(state_dict, strict=False)
    print(f"loaded_checkpoint missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")

    torch.manual_seed(args.seed)
    video = torch.rand(1, args.num_frames, 3, args.height, args.width, dtype=torch.float32)
    u = torch.rand(1, args.num_queries, dtype=torch.float32)
    v = torch.rand(1, args.num_queries, dtype=torch.float32)
    t_src = torch.randint(0, args.num_frames, (1, args.num_queries), dtype=torch.long)
    t_tgt = torch.randint(0, args.num_frames, (1, args.num_queries), dtype=torch.long)
    t_cam = torch.randint(0, args.num_frames, (1, args.num_queries), dtype=torch.long)
    aspect_ratio = torch.ones(1, 1, dtype=torch.float32)
    query = {"u": u, "v": v, "t_src": t_src, "t_tgt": t_tgt, "t_cam": t_cam}
    batch = {"video": video, "aspect_ratio": aspect_ratio, "query": query}

    with torch.no_grad():
        full = model(batch)
        memory = model.encode_video(video=video, aspect_ratio=aspect_ratio)
        split = model.decode_queries(video=video, query=query, memory=memory)

    print(f"memory_shape={tuple(memory.shape)}")
    max_abs = 0.0
    for name in OUTPUT_NAMES:
        diff = (full[name] - split[name]).abs()
        item = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, item)
        print(f"{name}: max_abs_diff={item:.9g}")
    print(f"overall_max_abs_diff={max_abs:.9g}")
    if max_abs != 0.0:
        raise SystemExit("Split forward is not bit-exact with full forward.")


if __name__ == "__main__":
    main()