# OpenD4RT MNN/Vulkan Runtime Pipeline

This package runs the checkpoint-converted OpenD4RT split model without loading the original PyTorch checkpoint.

## What It Does

Input video -> resize/sample to 32 frames at 256x256 -> MNN encoder -> MNN decoder in q8 batches -> 4D point cloud/tracks -> Viser demo package.

The current exported models are static for:

- frames: 32
- image size: 256x256
- encoder memory tokens: 4097
- decoder query batch: 8

## Validation Status

The following checks have already been completed for the current exported artifacts:

- Checkpoint-to-model compatibility was verified for `OpenD4RT_32CLIP_9Dataset_NoAUG`: `matched=627`, `missing=0`, `unexpected=0`, `shape_mismatch=0`.
- The PyTorch full model and split encoder/decoder path were compared on a small test shape, with output diffs equal to 0.
- The OpenD4RT checkpoint-backed encoder and decoder were exported to ONNX and converted to MNN.
- The original-size encoder was converted for `32 frames / 256x256`; its large weights are stored externally in `opend4rt_32clip_encoder_t32_256.mnn.weight`.
- The original-size decoder was converted as `artifacts/mnn_vulkan/decoder_mem4097_q8/opend4rt_32clip_decoder_mem4097_q8.mnn`.
- A small split MNN pipeline was previously run through Windows MNN/Vulkan successfully.
- The current video inference C++ runner has been compiled successfully, and the Python orchestration script passes syntax checks.
- The decoder bundle was regenerated after a failed run showed that the old top-level decoder file was not self-contained.

The original-size Vulkan video path has not been fully executed on the current laptop because a previous full-size iGPU test caused a black-screen reboot. For first validation on a new PC, use the staged commands below.

## Files Needed On Another Windows PC

Keep these files relative to the Open-d4rt repo root:

```text
artifacts/mnn_vulkan/MNN.dll
artifacts/mnn_vulkan/mnn_opend4rt_video_infer.exe
artifacts/mnn_vulkan/opend4rt_32clip_encoder_t32_256.mnn
artifacts/mnn_vulkan/opend4rt_32clip_encoder_t32_256.mnn.weight
artifacts/mnn_vulkan/decoder_mem4097_q8/opend4rt_32clip_decoder_mem4097_q8.mnn
scripts/infer_mnn_video_to_vis.py
vis/serve_demo_viser.py
requirements_mnn_vulkan.txt
```

For rebuilding the runner on that PC, also keep:

```text
scripts/mnn_opend4rt_video_infer.cpp
scripts/build_win_mnn_opend4rt_video_infer.bat
```

Rebuilding requires Visual Studio C++ Build Tools and the MNN Windows Vulkan build. Running the already-built exe does not require the checkpoint.

## Python Runtime Dependencies

Install these in the Python environment used to run the pipeline:

```bash
pip install -r requirements_mnn_vulkan.txt
```

`ffmpeg` is optional but recommended. If it is not installed, the script falls back to OpenCV's mp4 writer output.

## Run Inference

For the smoothest first run on a new PC, start with a low query count on CPU:

```powershell
python scripts/infer_mnn_video_to_vis.py `
  --video inputs\input.mp4 `
  --output-dir outputs\mnn_demo_cpu `
  --forward-type 0 `
  --point-cols 8 `
  --point-rows 8 `
  --max-points 64 `
  --max-tracks 8
```

After CPU succeeds, test the same small workload on Vulkan:

```powershell
python scripts/infer_mnn_video_to_vis.py `
  --video inputs\input.mp4 `
  --output-dir outputs\mnn_demo_vk `
  --forward-type 7 `
  --point-cols 8 `
  --point-rows 8 `
  --max-points 64 `
  --max-tracks 8
```

Then increase point density, for example:

```powershell
python scripts/infer_mnn_video_to_vis.py `
  --video inputs\input.mp4 `
  --output-dir outputs\mnn_demo `
  --forward-type 7 `
  --point-cols 24 `
  --point-rows 24 `
  --max-points 576 `
  --max-tracks 48
```

The script writes:

```text
outputs/mnn_demo/assets/input_video.mp4
outputs/mnn_demo/assets/demo_data.json
outputs/mnn_demo/mnn_outputs.npz
outputs/mnn_demo/manifest.json
```

## Visualize

```powershell
python vis/serve_demo_viser.py --root outputs\mnn_demo --host 127.0.0.1 --port 8081
```

Open:

```text
http://127.0.0.1:8081
```

## Notes

- The decoder MNN is fixed at q8, so the Python script automatically pads and batches queries in groups of 8.
- The script uses one static 32-frame clip. Longer videos are uniformly sampled to 32 frames; shorter videos are padded by repeating the final frame.
- The current laptop iGPU previously rebooted under full-size Vulkan load. For this machine, prefer `--forward-type 0` or run Vulkan on a stronger PC/GPU.
- The checkpoint is not needed after conversion. The large external encoder weight file must stay next to the encoder `.mnn` file.
- Use the decoder under `artifacts/mnn_vulkan/decoder_mem4097_q8/`; the old top-level decoder file should be treated as obsolete.