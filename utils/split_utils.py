# utils/split_utils.py
import os
from typing import Dict, List, Optional

# ---- basic text reading ----
def _read_int_list(txt_path: str) -> Optional[List[int]]:
    """
    Reads a list of integers (one per line or 'i,j,...') from txt.
    Returns None if file missing or empty.
    """
    if not os.path.isfile(txt_path):
        return None
    vals: List[int] = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow 'i' or 'i,j, ...'; we keep the first integer
            tok = s.split(",")[0]
            if tok.lstrip("-").isdigit():
                vals.append(int(tok))
    return sorted(list(set(vals))) if vals else None

def find_splits_dir(colmap_dir: str) -> Optional[str]:
    """
    Given a path like .../<video>/colmap, returns .../<video>/splits if it exists.
    """
    parent = os.path.dirname(colmap_dir.rstrip("/"))
    candidate = os.path.join(parent, "split")
    return candidate if os.path.isdir(candidate) else None

def load_splits_if_available(colmap_dir: str) -> Optional[Dict[str, List[int]]]:
    """
    Returns a dict of index lists (relative to the *sorted-by-image_name* order):
      {
        "train":        [indices] or [],
        "eval_static":  [indices] or [],
        "eval_dynamic": [indices] or []
      }
    Returns None if no usable split files are found.
    """
    splits_dir = find_splits_dir(colmap_dir)
    if splits_dir is None:
        return None

    train_txt  = os.path.join(splits_dir, "training_frames.txt")
    static_txt = os.path.join(splits_dir, "static_eval_frames.txt")
    dyn_txt    = os.path.join(splits_dir, "dynamic_eval_frames.txt")

    train  = _read_int_list(train_txt)
    stat   = _read_int_list(static_txt)
    dyn    = _read_int_list(dyn_txt)

    if (train is None) and (stat is None) and (dyn is None):
        return None

    return {
        "train":        train or [],
        "eval_static":  stat or [],
        "eval_dynamic": dyn or []
    }

def select_by_indices(cameras: List, indices: List[int]) -> List:
    """
    Given a list of cameras (already sorted by image_name) and a list of integer
    indices, returns the sublist while gracefully skipping OOB indices.
    """
    n = len(cameras)
    out = []
    for i in indices:
        if 0 <= i < n:
            out.append(cameras[i])
    return out
