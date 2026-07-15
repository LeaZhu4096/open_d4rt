"""Convert an ONNX model to MNN using MNNConvert."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--mnn", required=True)
    parser.add_argument("--mnnconvert", default=None)
    parser.add_argument("--biz-code", default="opend4rt_mnn_vulkan")
    args = parser.parse_args()

    converter = args.mnnconvert or shutil.which("MNNConvert") or shutil.which("mnnconvert")
    if converter is None:
        raise FileNotFoundError("Could not find MNNConvert/mnnconvert. Pass --mnnconvert.")

    onnx_path = Path(args.onnx)
    mnn_path = Path(args.mnn)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    mnn_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        converter,
        "-f",
        "ONNX",
        "--modelFile",
        onnx_path.as_posix(),
        "--MNNModel",
        mnn_path.as_posix(),
        "--bizCode",
        args.biz_code,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Converted: {mnn_path}")


if __name__ == "__main__":
    main()