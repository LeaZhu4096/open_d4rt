"""Export OpenD4RT to ONNX for MNN/Vulkan validation.

This script keeps the OpenD4RT model code as the source of truth. It can export
an exact model.yaml + checkpoint pair, or a tiny smoke profile that exercises the
same OpenD4RT modules before attempting the full checkpoint-sized graph.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.builder import build_model
from src.core.checkpoint import load_checkpoint

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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_smoke_profile(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    model = cfg["model"]
    model["input"]["clip_frames"] = args.num_frames
    model["input"]["image_size"] = [args.height, args.width]
    model["input"]["use_aspect_ratio_token"] = True

    enc = model["encoder"]
    enc["variant"] = "smoke"
    enc["num_layers"] = args.smoke_layers
    enc["hidden_dim"] = args.smoke_hidden_dim
    enc["num_heads"] = args.smoke_heads
    enc["mlp_ratio"] = 2.0
    enc["max_tokens"] = args.smoke_max_tokens
    enc["patch_size_t_h_w"] = [1, 8, 8]
    enc["pretrained"] = {"enabled": False, "path": "", "strict": False, "must_succeed": False}

    dec = model["decoder"]
    dec["num_layers"] = args.smoke_layers
    dec["hidden_dim"] = args.smoke_hidden_dim
    dec["num_heads"] = args.smoke_heads
    dec["mlp_ratio"] = 2.0

    model["query_embedding"]["local_rgb_patch"]["patch_size"] = args.smoke_query_patch_size
    return cfg


class OpenD4RTOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        video: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        aspect_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {
            "video": video,
            "aspect_ratio": aspect_ratio,
            "query": {
                "u": u,
                "v": v,
                "t_src": t_src,
                "t_tgt": t_tgt,
                "t_cam": t_cam,
            },
        }
        outputs = self.model(batch)
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def build_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(args.seed)
    video = torch.rand(args.batch_size, args.num_frames, 3, args.height, args.width, dtype=torch.float32)
    u = torch.rand(args.batch_size, args.num_queries, dtype=torch.float32)
    v = torch.rand(args.batch_size, args.num_queries, dtype=torch.float32)
    t_src = torch.randint(0, args.num_frames, (args.batch_size, args.num_queries), dtype=torch.long)
    t_tgt = torch.randint(0, args.num_frames, (args.batch_size, args.num_queries), dtype=torch.long)
    t_cam = torch.randint(0, args.num_frames, (args.batch_size, args.num_queries), dtype=torch.long)
    aspect_ratio = torch.ones(args.batch_size, 1, dtype=torch.float32)
    return video, u, v, t_src, t_tgt, t_cam, aspect_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default=None, help="Optional OpenD4RT checkpoint to bake into the ONNX/MNN weights.")
    parser.add_argument("--output", default="artifacts/mnn_vulkan/opend4rt.onnx")
    parser.add_argument("--profile", choices=["config", "smoke"], default="config")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--smoke-hidden-dim", type=int, default=64)
    parser.add_argument("--smoke-layers", type=int, default=1)
    parser.add_argument("--smoke-heads", type=int, default=4)
    parser.add_argument("--smoke-max-tokens", type=int, default=512)
    parser.add_argument("--smoke-query-patch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.model_config))

    if args.profile == "smoke":
        args.num_frames = args.num_frames or 2
        args.height = args.height or 32
        args.width = args.width or 32
        cfg = apply_smoke_profile(cfg, args)
        if args.ckpt_path:
            raise ValueError("--profile smoke changes model shapes and cannot load OpenD4RT checkpoints.")
    else:
        model_input = cfg["model"]["input"]
        image_size = model_input.get("image_size", [256, 256])
        args.num_frames = args.num_frames or int(model_input.get("clip_frames", 32))
        args.height = args.height or int(image_size[0])
        args.width = args.width or int(image_size[1])

    model = build_model(cfg["model"]).eval()
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = unwrap_state_dict(torch.load(Path(ckpt_path), map_location="cpu", mmap=True))
        if not state_dict:
            raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
        result = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"missing_keys={len(result.missing_keys)} unexpected_keys={len(result.unexpected_keys)}")

    wrapper = OpenD4RTOnnxWrapper(model).eval()
    inputs = build_inputs(args)
    with torch.no_grad():
        outputs = wrapper(*inputs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["video", "u", "v", "t_src", "t_tgt", "t_cam", "aspect_ratio"],
        output_names=OUTPUT_NAMES,
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_path.as_posix())
        onnx.checker.check_model(onnx_model)

    print(f"Exported: {output_path}")
    for name, tensor in zip(OUTPUT_NAMES, outputs):
        print(f"{name}: shape={tuple(tensor.shape)} mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")


if __name__ == "__main__":
    main()