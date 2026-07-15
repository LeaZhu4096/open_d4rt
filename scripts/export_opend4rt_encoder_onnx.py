"""Export checkpoint-backed OpenD4RT encoder + memory_proj subgraph to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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


class OpenD4RTEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        return self.model.encode_video(video=video, aspect_ratio=aspect_ratio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--output", default="artifacts/mnn_vulkan/opend4rt_32clip_encoder_t2_32.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"]).eval()
    state_dict = unwrap_state_dict(torch.load(Path(args.ckpt_path), map_location="cpu", mmap=True))
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.ckpt_path}")
    print(f"missing_keys={len(result.missing_keys)} unexpected_keys={len(result.unexpected_keys)}")

    torch.manual_seed(args.seed)
    video = torch.rand(args.batch_size, args.num_frames, 3, args.height, args.width, dtype=torch.float32)
    aspect_ratio = torch.ones(args.batch_size, 1, dtype=torch.float32)
    wrapper = OpenD4RTEncoderWrapper(model).eval()
    with torch.no_grad():
        memory = wrapper(video, aspect_ratio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (video, aspect_ratio),
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["video", "aspect_ratio"],
        output_names=["memory"],
    )

    if args.check:
        import onnx

        onnx.checker.check_model(output_path.as_posix())

    print(f"Exported: {output_path}")
    print(f"memory: shape={tuple(memory.shape)} mean={memory.mean().item():.6f} std={memory.std().item():.6f}")


if __name__ == "__main__":
    main()