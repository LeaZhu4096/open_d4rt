"""Export checkpoint-backed OpenD4RT decoder/query/head subgraph to ONNX."""

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


class OpenD4RTDecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.query_embedder = model.query_embedder
        self.decoder = model.decoder
        self.heads = model.heads

    def forward(
        self,
        memory: torch.Tensor,
        video: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_tokens = self.query_embedder(video=video, u=u, v=v, t_src=t_src, t_tgt=t_tgt, t_cam=t_cam)
        decoded = self.decoder(query_tokens, memory)
        outputs = self.heads(decoded)
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--output", default="artifacts/mnn_vulkan/opend4rt_32clip_decoder_q8.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"]).eval()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    state_dict = unwrap_state_dict(torch.load(Path(ckpt_path), map_location="cpu", mmap=True))
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"missing_keys={len(result.missing_keys)} unexpected_keys={len(result.unexpected_keys)}")

    dec_hidden = int(cfg["model"]["decoder"]["hidden_dim"])
    torch.manual_seed(args.seed)
    memory = torch.rand(args.batch_size, args.memory_tokens, dec_hidden, dtype=torch.float32)
    video = torch.rand(args.batch_size, args.num_frames, 3, args.height, args.width, dtype=torch.float32)
    u = torch.rand(args.batch_size, args.num_queries, dtype=torch.float32)
    v = torch.rand(args.batch_size, args.num_queries, dtype=torch.float32)
    t_src = torch.randint(0, args.num_frames, (args.batch_size, args.num_queries), dtype=torch.long)
    t_tgt = torch.randint(0, args.num_frames, (args.batch_size, args.num_queries), dtype=torch.long)
    t_cam = torch.randint(0, args.num_frames, (args.batch_size, args.num_queries), dtype=torch.long)

    wrapper = OpenD4RTDecoderWrapper(model).eval()
    with torch.no_grad():
        outputs = wrapper(memory, video, u, v, t_src, t_tgt, t_cam)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (memory, video, u, v, t_src, t_tgt, t_cam),
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["memory", "video", "u", "v", "t_src", "t_tgt", "t_cam"],
        output_names=OUTPUT_NAMES,
    )

    if args.check:
        import onnx

        onnx.checker.check_model(output_path.as_posix())

    print(f"Exported: {output_path}")
    for name, tensor in zip(OUTPUT_NAMES, outputs):
        print(f"{name}: shape={tuple(tensor.shape)} mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")


if __name__ == "__main__":
    main()