#!/usr/bin/env python3
"""Run an OpenD4RT MNN split model on one video and export a Viser demo package.

This is the portable PC runtime path: it does not load OpenD4RT checkpoints.
It uses encoder.mnn + encoder.mnn.weight + decoder.mnn through a small C++ MNN runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for video IO. Install it with: pip install opencv-python") from exc
    return cv2


def _load_video_rgb(path: Path, max_frames: int = 0) -> tuple[np.ndarray, float]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 10.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(frames) >= int(max_frames):
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), fps


def _resize_video(video_rgb: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    cv2 = _require_cv2()
    h, w = image_hw
    resized = [
        cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[0] >= h else cv2.INTER_LINEAR)
        for frame in video_rgb
    ]
    return np.stack(resized, axis=0)


def _grid_query_points(width: int, height: int, cols: int, rows: int, margin_ratio: float, max_points: int) -> np.ndarray:
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    margin_x = float(max(width - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    margin_y = float(max(height - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    xs = np.linspace(margin_x, float(max(width - 1, 0)) - margin_x, num=cols, dtype=np.float32)
    ys = np.linspace(margin_y, float(max(height - 1, 0)) - margin_y, num=rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if grid.shape[0] > max_points:
        pick = np.linspace(0, grid.shape[0] - 1, num=max_points, dtype=np.int64)
        grid = grid[pick]
    return grid.astype(np.float32)


def parse_args() -> argparse.Namespace:
    default_artifacts = REPO_ROOT / "artifacts" / "mnn_vulkan"
    parser = argparse.ArgumentParser(description="Video -> OpenD4RT MNN -> 4D point cloud -> Viser package.")
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output demo package directory.")
    parser.add_argument("--encoder", type=Path, default=default_artifacts / "opend4rt_32clip_encoder_t32_256.mnn")
    parser.add_argument(
        "--decoder",
        type=Path,
        default=default_artifacts / "decoder_mem4097_q2048" / "opend4rt_32clip_decoder_mem4097_q2048.mnn",
    )
    parser.add_argument("--runner", type=Path, default=default_artifacts / "mnn_opend4rt_video_infer.exe")
    parser.add_argument("--build-runner", action="store_true", help="Build the C++ runner before inference on Windows.")
    parser.add_argument("--forward-type", type=int, default=7, help="MNN forward type. 7=Vulkan, 0=CPU.")
    parser.add_argument("--num-frames", type=int, default=32, help="Static frame count used by the exported encoder.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--query-batch", type=int, default=2048, help="Static decoder query count in the exported decoder.")
    parser.add_argument("--point-cols", type=int, default=64)
    parser.add_argument("--point-rows", type=int, default=64)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--max-tracks", type=int, default=500)
    parser.add_argument("--track-source-frame", type=int, default=0)
    parser.add_argument("--keep-runtime-io", action="store_true", help="Keep raw binary files used to call the C++ runner.")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        return arr.astype(bool).tolist()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64).tolist()
    if np.issubdtype(arr.dtype, np.floating):
        cleaned = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.round(cleaned, 6).tolist()
    return arr.tolist()


def _sigmoid_bool(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))) > 0.5


def _write_video_assets(video_rgb: np.ndarray, fps: float, assets_dir: Path) -> None:
    cv2 = _require_cv2()
    assets_dir.mkdir(parents=True, exist_ok=True)
    poster = assets_dir / "video_poster.jpg"
    cv2.imwrite(str(poster), np.asarray(video_rgb[0], dtype=np.uint8)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    out_path = assets_dir / "input_video.mp4"
    temp_path = assets_dir / "_tmp_input_video.mp4"
    h, w = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(fps, 1.0)), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {temp_path}")
    try:
        for frame in video_rgb:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-i", str(temp_path), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            shutil.copy2(temp_path, out_path)
    else:
        shutil.copy2(temp_path, out_path)
    temp_path.unlink(missing_ok=True)


def _pad_or_sample_video(video_rgb: np.ndarray, num_frames: int) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    if video.shape[0] == num_frames:
        return video
    if video.shape[0] > num_frames:
        ids = np.linspace(0, video.shape[0] - 1, num=num_frames, dtype=np.int64)
        return video[ids]
    pad = np.repeat(video[-1:,...], num_frames - video.shape[0], axis=0)
    return np.concatenate([video, pad], axis=0)


def _build_queries(point_uv_norm: np.ndarray, track_uv_norm: np.ndarray, num_frames: int, track_src: int) -> tuple[dict[str, np.ndarray], dict[str, slice]]:
    p = int(point_uv_norm.shape[0])
    tr = int(track_uv_norm.shape[0])
    target_ids = np.arange(num_frames, dtype=np.int32)

    point_uv = np.tile(point_uv_norm.astype(np.float32), (num_frames, 1))
    point_t = np.repeat(target_ids, p)

    track_uv = np.repeat(track_uv_norm.astype(np.float32), num_frames, axis=0)
    track_src_arr = np.full((tr * num_frames,), int(np.clip(track_src, 0, num_frames - 1)), dtype=np.int32)
    track_tgt = np.tile(target_ids, tr)

    u = [point_uv[:, 0], track_uv[:, 0], track_uv[:, 0]]
    v = [point_uv[:, 1], track_uv[:, 1], track_uv[:, 1]]
    t_src = [point_t, track_src_arr, track_src_arr]
    t_tgt = [point_t, track_tgt, track_tgt]
    t_cam = [np.zeros_like(point_t), track_tgt.copy(), np.zeros_like(track_tgt)]

    sizes = [p * num_frames, tr * num_frames, tr * num_frames]
    s0 = slice(0, sizes[0])
    s1 = slice(s0.stop, s0.stop + sizes[1])
    s2 = slice(s1.stop, s1.stop + sizes[2])
    return {
        "u": np.concatenate(u).astype(np.float32),
        "v": np.concatenate(v).astype(np.float32),
        "t_src": np.concatenate(t_src).astype(np.int32),
        "t_tgt": np.concatenate(t_tgt).astype(np.int32),
        "t_cam": np.concatenate(t_cam).astype(np.int32),
    }, {"point": s0, "track_local": s1, "track_ref": s2}


def _sample_rgb(video_rgb: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    pts = np.asarray(uv_px, dtype=np.float32)
    out = np.zeros((video.shape[0], pts.shape[0], 3), dtype=np.uint8)
    for ti in range(video.shape[0]):
        for qi, (x, y) in enumerate(pts):
            xi = int(np.clip(round(float(x)), 0, video.shape[2] - 1))
            yi = int(np.clip(round(float(y)), 0, video.shape[1] - 1))
            out[ti, qi] = video[ti, yi, xi]
    return out


def _estimate_bounds(xyz: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(xyz).all(axis=-1) & np.asarray(vis, dtype=bool)
    if np.any(valid):
        pts = np.asarray(xyz, dtype=np.float32)[valid]
        xyz_min = pts.min(axis=0).astype(np.float32)
        xyz_max = pts.max(axis=0).astype(np.float32)
        center = ((xyz_min + xyz_max) * 0.5).astype(np.float32)
        radius = np.asarray([float(max(np.max(xyz_max - xyz_min) * 0.55, 1e-3))], dtype=np.float32)
        return xyz_min, xyz_max, center, radius
    z = np.zeros((3,), dtype=np.float32)
    return z, z, z, np.asarray([1.0], dtype=np.float32)


def _run_runner(args: argparse.Namespace, runtime_dir: Path, total_queries: int) -> None:
    if args.build_runner:
        build_script = REPO_ROOT / "scripts" / "build_win_mnn_opend4rt_video_infer.bat"
        subprocess.run([str(build_script)], cwd=str(REPO_ROOT), check=True)
    _require_file(args.runner, "MNN video runner")
    _require_file(args.encoder, "encoder MNN")
    _require_file(args.decoder, "decoder MNN")
    weight = args.encoder.with_suffix(args.encoder.suffix + ".weight")
    _require_file(weight, "encoder MNN external weight")

    cmd = [
        str(args.runner),
        str(args.encoder),
        str(args.decoder),
        str(runtime_dir / "input"),
        str(runtime_dir / "output"),
        str(total_queries),
        str(args.forward_type),
        str(args.num_frames),
        str(args.height),
        str(args.width),
        str(args.query_batch),
        "4097",
        "1280",
    ]
    env = os.environ.copy()
    env["PATH"] = str(args.runner.parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        if result.returncode == 3221225477:
            raise RuntimeError(
                "MNN native runner crashed with Windows access violation 0xC0000005. "
                "This usually indicates a crash inside the selected MNN backend or GPU driver. "
                "Check the last [mnn-video] stage above to see whether it happened during encoder or decoder execution."
            )
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)


def main() -> int:
    args = parse_args()
    _require_file(args.video, "input video")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.output_dir / "runtime_io"
    input_dir = runtime_dir / "input"
    output_dir = runtime_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_video, fps = _load_video_rgb(args.video, max_frames=0)
    clip_video = _pad_or_sample_video(raw_video, args.num_frames)
    video_rgb = _resize_video(clip_video, (args.height, args.width)).astype(np.uint8)
    video_nchw = (video_rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)[None]
    aspect = np.asarray([float(args.width) / float(max(1, args.height))], dtype=np.float32)

    point_uv_px = _grid_query_points(args.width, args.height, args.point_cols, args.point_rows, 0.02, args.max_points)
    track_count = int(min(args.max_tracks, point_uv_px.shape[0]))
    track_ids = np.linspace(0, point_uv_px.shape[0] - 1, num=max(1, track_count), dtype=np.int64)
    track_uv_px = point_uv_px[track_ids]

    point_uv_norm = point_uv_px.copy().astype(np.float32)
    point_uv_norm[:, 0] /= float(max(args.width - 1, 1))
    point_uv_norm[:, 1] /= float(max(args.height - 1, 1))
    track_uv_norm = track_uv_px.copy().astype(np.float32)
    track_uv_norm[:, 0] /= float(max(args.width - 1, 1))
    track_uv_norm[:, 1] /= float(max(args.height - 1, 1))

    queries, slices = _build_queries(point_uv_norm, track_uv_norm, args.num_frames, args.track_source_frame)
    total_queries = int(queries["u"].shape[0])

    video_nchw.astype(np.float32).tofile(input_dir / "video_f32.bin")
    aspect.astype(np.float32).tofile(input_dir / "aspect_ratio_f32.bin")
    queries["u"].tofile(input_dir / "query_u_f32.bin")
    queries["v"].tofile(input_dir / "query_v_f32.bin")
    queries["t_src"].tofile(input_dir / "query_t_src_i32.bin")
    queries["t_tgt"].tofile(input_dir / "query_t_tgt_i32.bin")
    queries["t_cam"].tofile(input_dir / "query_t_cam_i32.bin")

    _run_runner(args, runtime_dir, total_queries)

    xyz = np.fromfile(output_dir / "xyz_3d_f32.bin", dtype=np.float32).reshape(total_queries, 3)
    uv = np.fromfile(output_dir / "uv_2d_f32.bin", dtype=np.float32).reshape(total_queries, 2)
    vis_logits = np.fromfile(output_dir / "visibility_f32.bin", dtype=np.float32).reshape(total_queries)
    conf = np.fromfile(output_dir / "confidence_f32.bin", dtype=np.float32).reshape(total_queries)

    p = point_uv_px.shape[0]
    tr = track_uv_px.shape[0]
    t = args.num_frames
    point_slice = slices["point"]
    track_local_slice = slices["track_local"]
    track_ref_slice = slices["track_ref"]

    point_xyz = xyz[point_slice].reshape(t, p, 3)
    point_conf = conf[point_slice].reshape(t, p)
    point_vis = np.isfinite(point_xyz).all(axis=-1) & _sigmoid_bool(vis_logits[point_slice].reshape(t, p))
    point_uv_px_seq = np.tile(point_uv_px[None, :, :], (t, 1, 1)).astype(np.float32)
    point_rgb = _sample_rgb(video_rgb, point_uv_px)

    track_xyz_ref0 = xyz[track_ref_slice].reshape(tr, t, 3)
    track_uv_norm_pred = uv[track_local_slice].reshape(tr, t, 2)
    track_uv_px_pred = track_uv_norm_pred.copy()
    track_uv_px_pred[..., 0] *= float(max(args.width - 1, 1))
    track_uv_px_pred[..., 1] *= float(max(args.height - 1, 1))
    track_vis = np.isfinite(track_xyz_ref0).all(axis=-1) & _sigmoid_bool(vis_logits[track_local_slice].reshape(tr, t))
    track_conf = conf[track_local_slice].reshape(tr, t)

    motion = np.linalg.norm(point_xyz - point_xyz[:1], axis=-1)
    motion_score = np.nan_to_num(np.nanpercentile(motion, 90, axis=0).astype(np.float32), nan=0.0)
    dyn_thresh = float(np.nanpercentile(motion_score, 80)) if np.any(motion_score > 0) else math.inf
    point_is_dynamic = motion_score >= dyn_thresh
    bounds_min, bounds_max, bounds_center, bounds_radius = _estimate_bounds(point_xyz, point_vis)

    focal = float(max(args.width, args.height))
    ref0_k = np.asarray([[focal, 0.0, (args.width - 1) * 0.5], [0.0, focal, (args.height - 1) * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

    data = {
        "meta": {
            "videoWidth": int(args.width),
            "videoHeight": int(args.height),
            "numFrames": int(t),
            "fps": float(fps),
            "clipFrames": int(t),
            "bounds": {
                "min": _jsonable(bounds_min),
                "max": _jsonable(bounds_max),
                "center": _jsonable(bounds_center),
                "radius": float(bounds_radius.reshape(-1)[0]),
            },
            "ref0K": _jsonable(ref0_k),
            "runtime": "mnn_split_vulkan" if args.forward_type == 7 else "mnn_split_cpu",
            "encoder": str(args.encoder),
            "decoder": str(args.decoder),
            "queryBatch": int(args.query_batch),
            "totalQueries": int(total_queries),
            "sourceVideo": str(args.video),
            "camera": None,
            "cameraPred": None,
        },
        "points": {
            "queryUvPx": _jsonable(point_uv_px),
            "xyzRef0": _jsonable(point_xyz),
            "visibility": _jsonable(point_vis),
            "uvPx": _jsonable(point_uv_px_seq),
            "confidence": _jsonable(point_conf),
            "motionScore": _jsonable(motion_score),
            "isDynamic": _jsonable(point_is_dynamic),
            "rgb": _jsonable(point_rgb),
        },
        "tracks": {
            "queryUvPx": _jsonable(track_uv_px),
            "queryTSrc": _jsonable(np.full((tr,), int(args.track_source_frame), dtype=np.int64)),
            "xyzRef0": _jsonable(track_xyz_ref0),
            "uvPx": _jsonable(track_uv_px_pred),
            "visibility": _jsonable(track_vis),
            "confidence": _jsonable(track_conf),
            "stitchDiagnostics": {"mode": "single_clip_mnn", "clipFrames": int(t), "chunks": []},
        },
        "tracksGt": None,
    }
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "demo_data.json").write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    _write_video_assets(video_rgb, fps, assets_dir)

    np.savez_compressed(
        args.output_dir / "mnn_outputs.npz",
        point_xyz_ref0=point_xyz,
        point_visibility=point_vis,
        point_confidence=point_conf,
        track_xyz_ref0=track_xyz_ref0,
        track_uv_px=track_uv_px_pred,
        track_visibility=track_vis,
        track_confidence=track_conf,
        point_query_uv_px=point_uv_px,
        track_query_uv_px=track_uv_px,
    )
    manifest = {
        "entry": "vis/serve_demo_viser.py --root <this directory>",
        "video": "assets/input_video.mp4",
        "data_json": "assets/demo_data.json",
        "npz": "mnn_outputs.npz",
        "portable_files_needed": [
            "artifacts/mnn_vulkan/opend4rt_32clip_encoder_t32_256.mnn",
            "artifacts/mnn_vulkan/opend4rt_32clip_encoder_t32_256.mnn.weight",
            "artifacts/mnn_vulkan/decoder_mem4097_q2048/opend4rt_32clip_decoder_mem4097_q2048.mnn",
            "artifacts/mnn_vulkan/mnn_opend4rt_video_infer.exe",
            "artifacts/mnn_vulkan/MNN.dll",
            "scripts/infer_mnn_video_to_vis.py",
            "vis/serve_demo_viser.py",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    if not args.keep_runtime_io:
        shutil.rmtree(runtime_dir, ignore_errors=True)

    print(f"Saved MNN Viser package: {args.output_dir}")
    print(f"Serve with: python vis/serve_demo_viser.py --root {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
