import os, json, subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Matplotlib only for colorizing to PNG; runs headless with Agg.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.image_utils import psnr_map

def _to_3ch(img: Image.Image) -> Image.Image:
    # Ensure RGB (drop A if RGBA)
    if img.mode == "RGB":
        return img
    return img.convert("RGB")

def generate_psnr_heatmaps_for_folder(
    folder_path: str,
    out_subdir: str = "psnr_heatmaps",
    vmin: float = 10.0,
    vmax: float = 45.0,
    error_colors: bool = True,
    make_video: bool = True,
    fps: int = 15,
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    pattern_sort_key=None,
    rotate_ccw_90: bool = False,
    save_legend: bool = True,
    add_colorbar_per_frame: bool = False
):
    """
    Generate per-pixel PSNR heatmaps for a folder with:
        <folder_path>/renders/*.png
        <folder_path>/gt/*.png

    Saves colorized frames to <folder_path>/<out_subdir> and optionally stitches
    them into an MP4. Optionally rotates outputs by 90° CCW and adds a legend.

    Color convention (when error_colors=True):
        - Red  = low PSNR (high error)
        - Blue = high PSNR (low error)

    Args:
        folder_path: Path to a folder containing 'renders/' and 'gt/' subfolders.
        out_subdir: Name of subfolder to write heatmaps into.
        vmin, vmax: PSNR range (dB) mapped to the colormap.
        error_colors: If True, red = low PSNR and blue = high PSNR (by reversing colormap).
        make_video: If True, tries to create psnr_heatmap.mp4 with ffmpeg.
        fps: Frames per second for the MP4.
        device: 'cuda' or 'cpu' for tensor ops.
        pattern_sort_key: Optional key function for sorting filenames.
        rotate_ccw_90: If True, rotate saved frames 90 degrees counter-clockwise.
        save_legend: If True, save a standalone legend PNG.
        add_colorbar_per_frame: If True, embed a colorbar on every frame (slower).

    Returns:
        pathlib.Path to the output directory with PNGs (and MP4 if created).
    """
    
    def _colorize_psnr_array(pmap_hw: np.ndarray,
                             vmin: float,
                             vmax: float,
                             error_colors: bool,
                             cmap_name: str = "turbo") -> Image.Image:
        """
        Colorize a single [H,W] PSNR array into an RGB heatmap image.
        If error_colors=True: red = low PSNR, blue = high PSNR.
        NaNs (masked pixels) are colored black.
        """
        arr = np.array(pmap_hw, dtype=np.float32, copy=True)
        mask = ~np.isfinite(arr)
        # Replace NaNs with vmin for normalization; they'll still be drawn as black via masked array.
        arr[mask] = vmin
        arr = np.clip(arr, vmin, vmax)

        # Normalize to [0,1]. Invert if we want red for low PSNR.
        if error_colors:
            norm = (vmax - arr) / max(vmax - vmin, 1e-6)
        else:
            norm = (arr - vmin) / max(vmax - vmin, 1e-6)

        norm = np.ma.masked_array(norm, mask=mask)

        cmap = plt.get_cmap(cmap_name)
        cmap.set_bad(color=(0, 0, 0, 1))  # masked -> black
        rgba = cmap(norm)                 # [H,W,4]
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        return Image.fromarray(rgb)

    def _save_colorbar_legend(out_dir: Path,
                              vmin: float,
                              vmax: float,
                              error_colors: bool,
                              cmap_name: str = "turbo",
                              orientation: str = "horizontal",
                              filename: str = "psnr_colorbar_legend.png") -> Path:
        """
        Save a standalone colorbar legend describing the PSNR color mapping.
        """
        cmap = plt.get_cmap(cmap_name)
        if error_colors:
            cmap = cmap.reversed()  # red for low values

        norm = Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(
            figsize=(4.0 if orientation == "horizontal" else 1.2,
                     1.0 if orientation == "horizontal" else 3.2),
            dpi=200
        )
        fig.subplots_adjust(
            bottom=0.35 if orientation == "horizontal" else 0.1,
            top=0.85 if orientation == "horizontal" else 0.9,
            left=0.1, right=0.9
        )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, orientation=orientation)
        cbar.set_label(
            "Per-pixel PSNR (dB)\nRed = low PSNR (high error) • Blue = high PSNR (low error)",
            fontsize=8
        )
        ax.axis("off")

        out_path = out_dir / filename
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return out_path

    folder = Path(folder_path)
    renders_dir = folder / "renders"
    gt_dir = folder / "gt"
    out_dir = folder / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not renders_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Expected subfolders 'renders' and 'gt' in {folder_path}")

    # Optional: save a single legend image
    if save_legend:
        _save_colorbar_legend(
            out_dir=out_dir,
            vmin=vmin,
            vmax=vmax,
            error_colors=error_colors,
            cmap_name="turbo",
            orientation="horizontal",
            filename="psnr_colorbar_legend.png"
        )

    fnames = sorted(os.listdir(renders_dir))
    if pattern_sort_key is not None:
        fnames = sorted(fnames, key=pattern_sort_key)

    mapping = []

    for idx, fname in enumerate(fnames):
        # Load and ensure RGB
        r_img = Image.open(renders_dir / fname)
        g_img = Image.open(gt_dir / fname)
        if r_img.mode != "RGB":
            r_img = r_img.convert("RGB")
        if g_img.mode != "RGB":
            g_img = g_img.convert("RGB")

        # To torch tensors [1,3,H,W], normalized [0,1]
        r = TF.to_tensor(r_img).unsqueeze(0).to(device)[:, :3]
        g = TF.to_tensor(g_img).unsqueeze(0).to(device)[:, :3]

        # Compute per-pixel PSNR map: [1,1,H,W] (clamped to [vmin,vmax] to stabilize colors)
        pmap = psnr_map(r, g, mask=None, max_val=1.0, eps=1e-8, clamp_range=(vmin, vmax))
        pmap_hw = pmap[0, 0].detach().cpu().numpy()

        if add_colorbar_per_frame:
            # Use matplotlib to draw image + colorbar together
            cmap = plt.get_cmap("turbo")
            if error_colors:
                cmap = cmap.reversed()

            fig, ax = plt.subplots(dpi=200)
            im = ax.imshow(pmap_hw, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(
                "PSNR (dB)\nRed = low PSNR (high error) • Blue = high PSNR (low error)",
                fontsize=6
            )

            tmp_path = out_dir / f"{idx:05d}_psnr_tmp.png"
            fig.savefig(tmp_path, bbox_inches="tight", pad_inches=0, transparent=False)
            plt.close(fig)

            heat = Image.open(tmp_path).convert("RGB")
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Faster colorization without embedding a colorbar on the frame
            heat = _colorize_psnr_array(pmap_hw, vmin=vmin, vmax=vmax,
                                        error_colors=error_colors, cmap_name="turbo")

        # Rotate 90° CCW if requested
        if rotate_ccw_90:
            heat = heat.rotate(90, expand=True)

        out_path = out_dir / f"{idx:05d}_psnr.png"
        heat.save(out_path)
        mapping.append({"index": idx, "file": fname})

    # Save frame->filename mapping
    with open(out_dir / "_frame_map.json", "w") as f:
        json.dump(mapping, f, indent=2)

    # Optional: stitch to MP4 with ffmpeg
    if make_video:
        mp4_path = out_dir / "psnr_heatmap.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(out_dir / "%05d_psnr.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"[Warn] ffmpeg failed to create video: {e}\n"
                  f"You can run the following manually:\n"
                  f"{' '.join(cmd)}")

    return out_dir