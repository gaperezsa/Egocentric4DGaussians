# utils/depth_gt_utils.py
import os, re, json, shutil, subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import cv2
from PIL import Image

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _extract_frame_index(name: str) -> Optional[int]:
    """
    Extract a frame index from an image_name. We try common patterns:
      - ..._00012
      - frame_00012
      - 00012
    Returns int or None if not found.
    """
    # prefer last group of consecutive digits
    m = re.findall(r'(\d+)', name)
    if not m:
        return None
    return int(m[-1])

def _resize_nearest(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """size = (W, H)"""
    W, H = size
    return cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)

def _to_meters(arr: np.ndarray) -> np.ndarray:
    """
    Heuristic conversion to meters:
      - If values look like millimeters (max > 1000), divide by 1000.
      - If values are already small (<= 100), assume meters.
      - Clip negatives to 0.
    """
    if arr.size == 0:
        return arr.astype(np.float32)
    amax = float(np.nanmax(arr))
    out = arr.astype(np.float32)
    if amax > 900:  # looks like millimeters
        out = out / 1000.0
    out[out < 0] = 0.0
    return out

def _normalize01_for_vis(arr: np.ndarray) -> np.ndarray:
    m = float(np.max(arr)) if arr.size else 0.0
    if m <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / m).astype(np.float32)

def _true_depth_search_root(colmap_dir: str) -> Optional[Path]:
    """
    For a source_path like .../<video>/colmap, we look for sibling:
      .../<video>/sparse_unprocessed_gt_depth/
    """
    p = Path(colmap_dir).resolve()
    sib = p.parent / "sparse_unprocessed_gt_depth"
    return sib if sib.is_dir() else None

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def _decode_avi_to_frames(avi_path: Path, out_dir: Path, fps: Optional[int] = None):
    """
    Decode depth .avi to a folder of 16-bit PNG frames using ffmpeg when available.
    We keep vsync=0 to avoid frame duplication/dropping. If fps is None, we let ffmpeg choose.
    NOTE: Requires ffmpeg in PATH.
    """
    _ensure_dir(str(out_dir))
    fps_arg = [] if fps is None else ["-vf", f"fps=fps={int(fps)}"]
    cmd = ["ffmpeg", "-y", "-i", str(avi_path), "-vsync", "0"] + fps_arg + ["-f", "image2", str(out_dir / "%05d.png"), "-loglevel", "quiet"]
    subprocess.run(cmd, check=True)

def _load_all_frames_from_avi(avi_path: Path) -> List[np.ndarray]:
    """
    Fallback if ffmpeg is not available. Uses OpenCV VideoCapture.
    WARNING: many depth AVIs are encoded with non-standard codecs; prefer ffmpeg path above.
    """
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open AVI: {avi_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        # frame is usually 8-bit BGR; for true depth, this may be wrong.
        # We convert to grayscale; if your AVI is 16-bit packed, please use ffmpeg path instead.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
    cap.release()
    return frames

def _collect_true_depth_sources(root: Path) -> Tuple[str, List[Path]]:
    """
    Returns (mode, sources):
      - "avi":   [path_to_avi]
      - "pngs":  [sorted list of PNGs]
      - "tensors":[sorted list of .pt or .npy]
      - "none":  []
    Expected locations under root:
      root/depth_video.avi
      root/frames/*.png
      root/tensors/*.pt or *.npy
    """
    avi = root / "depth_video.avi"
    if avi.is_file():
        return "avi", [avi]

    frames_dir = root / "frames"
    if frames_dir.is_dir():
        pngs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() == ".png"])
        if pngs:
            return "pngs", pngs

    tens_dir = root / "tensors"
    if tens_dir.is_dir():
        tens = sorted([p for p in tens_dir.iterdir() if p.suffix.lower() in (".pt", ".npy")])
        if tens:
            return "tensors", tens

    return "none", []

def _load_tensor(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".pt":
        t = torch.load(path)
        while t.dim() > 2:
            t = t.squeeze(0)
        return t.cpu().numpy().astype(np.float32)
    elif path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported tensor file: {path}")

def _write_depth_pair(out_tensor: Path, out_vis: Path, depth_m: np.ndarray):
    out_tensor.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(depth_m.astype(np.float32)), out_tensor)
    vis = _normalize01_for_vis(depth_m)
    # to 3c uint8 PNG
    vis3 = np.repeat(vis[..., None], 3, axis=2)
    Image.fromarray((vis3 * 255.0).astype(np.uint8)).save(out_vis)

def ensure_true_depth_gt(
    run_dir: str,
    source_path: str,
    image_names: List[str],
    target_size: Tuple[int, int]
):
    """
    Ensure that run_dir has:
      run_dir/depth_gt_tensors/<image_name>.pt  (meters)
      run_dir/depth_gt/<image_name>.png         (visualization)
    built from the *true* sparse/unprocessed GT (NOT the dense training depth).

    It looks for .../<video>/sparse_unprocessed_gt_depth/{depth_video.avi | frames/*.png | tensors/*}
    and maps frames by the index parsed from image_name.

    If depth GT already present, it does nothing.
    """
    run = Path(run_dir)
    tgt_t = run / "depth_gt_tensors"
    tgt_v = run / "depth_gt"
    if (tgt_t.exists() and any(tgt_t.iterdir())):
        # already materialized
        return
    
    tgt_t.mkdir(parents=True, exist_ok=True)
    tgt_v.mkdir(parents=True, exist_ok=True)

    # Save a manifest so metrics can know source
    manifest_path = run / "render_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.load(open(manifest_path, "r"))
        except Exception:
            manifest = {}
    manifest.update({"source_path": source_path})
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # find raw sparse depth root
    true_root = _true_depth_search_root(source_path)
    if true_root is None:
        print(f"[ensure_true_depth_gt] No sparse_unprocessed_gt_depth folder next to {source_path}. Skipping true depth materialization.")
        return

    mode, sources = _collect_true_depth_sources(true_root)
    if mode == "none":
        print(f"[ensure_true_depth_gt] No true depth sources found under {true_root}. Skipping.")
        return

    print(f"[ensure_true_depth_gt] Using true depth mode: {mode}")

    # prepare cache if AVI
    avi_frames = None
    frames_dir_tmp = None
    if mode == "avi":
        avi_path = sources[0]
        # try ffmpeg → PNGs in a temp folder inside run_dir
        frames_dir_tmp = run / "_true_depth_frames_tmp"
        breakpoint()
        if _ffmpeg_available():
            try:
                _decode_avi_to_frames(avi_path, frames_dir_tmp)
                mode = "pngs"
                sources = sorted([p for p in frames_dir_tmp.iterdir() if p.suffix.lower()==".png"])
            except Exception as e:
                print(f"[ensure_true_depth_gt] ffmpeg decode failed: {e}. Falling back to OpenCV.")
        if mode == "avi":
            # fallback: load all frames (NOTE: may not be correct for 16-bit depth AVIs)
            avi_frames = _load_all_frames_from_avi(avi_path)

    # map index->source item
    idx2src = {}
    if mode == "pngs":
        for i, p in enumerate(sources):
            idx2src[i] = p
    elif mode == "tensors":
        for i, p in enumerate(sources):
            idx2src[i] = p
    elif mode == "avi" and avi_frames is not None:
        for i in range(len(avi_frames)):
            idx2src[i] = i  # store the int index, we pull from avi_frames list

    # build outputs
    W, H = target_size
    for name in image_names:
        idx = _extract_frame_index(name)
        if idx is None or idx not in idx2src:
            # no matching true depth frame; skip
            continue

        if mode == "pngs":
            arr = np.array(Image.open(idx2src[idx]), dtype=np.float32)
        elif mode == "tensors":
            arr = _load_tensor(idx2src[idx])
        elif mode == "avi":
            arr = avi_frames[idx]
        else:
            continue

        if arr.ndim == 3:
            # take single channel if it's multi-channel
            arr = arr[..., 0]

        # resize to RGB size
        arr = _resize_nearest(arr, (W, H))
        # to meters (heuristic)
        arr_m = _to_meters(arr)

        out_t = tgt_t / f"{name}.pt"
        out_v = tgt_v / f"{name}.png"
        _write_depth_pair(out_t, out_v, arr_m)

    # cleanup temp frames if created
    if frames_dir_tmp and frames_dir_tmp.exists():
        try:
            shutil.rmtree(frames_dir_tmp)
        except Exception:
            pass
