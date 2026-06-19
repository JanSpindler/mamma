#!/usr/bin/env python3
"""Convert THuman4.0 dataset to MAMMA-compatible layout.

Source:  anchor_data/anchor_thuman/<subject>/
           images/<cam>/<frame>.jpg
           calibration.json          (K, R, T per camera)

Output:  anchor_data/processed/<subject>/
           images/<cam>/             (copied from source)
           calibration.json          (MAMMA OpenCV-flat format)

Run:
    python scripts/convert_thuman.py

Then per subject:
    python -m inference run \\
        --cfg      configs/examples/presets/quick.yaml \\
        --footage  anchor_data/processed \\
        --seq_name subject00 \\
        --calib    anchor_data/processed/subject00/calibration.json \\
        --out-tag  thuman4 -v
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

SRC_ROOT = Path("/lustre/mlnvme/data/jspindle_hpc-anchor/anchor_thuman")
DST_ROOT = Path("/lustre/mlnvme/data/jspindle_hpc-anchor/processed")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def detect_image_size(images_src: Path) -> tuple[int, int]:
    """Read width, height from the first image found under images_src."""
    for cam_dir in sorted(images_src.iterdir()):
        if not cam_dir.is_dir():
            continue
        for img_path in sorted(cam_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                with PILImage.open(img_path) as img:
                    return img.width, img.height
    raise RuntimeError(f"No images found under {images_src}")


def convert_calibration(src_json: Path, image_size: tuple[int, int]) -> dict:
    """Convert THuman4 calibration.json to MAMMA OpenCV-flat format."""
    with open(src_json) as f:
        src = json.load(f)

    width, height = image_size
    out = {}
    for cam_name, c in src.items():
        K = np.array(c["K"], dtype=np.float64).reshape(3, 3)
        R = np.array(c["R"], dtype=np.float64).reshape(3, 3)
        T = np.array(c["T"], dtype=np.float64).reshape(3)
        # THuman4 convention: x_cam = R @ x_world + T
        # MAMMA expects [R | T] as a 3x4 extrinsics_matrix
        ext = np.hstack([R, T.reshape(3, 1)])
        out[cam_name] = {
            "intrinsic_matrix": K.tolist(),
            "extrinsics_matrix": ext.tolist(),
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [width, height],
        }
    return out


def copy_cam_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    for f in tqdm(files, desc=src.name, unit="img", leave=False):
        shutil.copy2(f, dst / f.name)


def process_subject(subject_dir: Path, dst_dir: Path) -> None:
    images_src = subject_dir / "images"
    calib_src = subject_dir / "calibration.json"

    if not images_src.is_dir():
        print(f"  [skip] no images/ dir found in {subject_dir}")
        return
    if not calib_src.is_file():
        print(f"  [skip] no calibration.json found in {subject_dir}")
        return

    image_size = detect_image_size(images_src)
    print(f"  detected image size: {image_size[0]}x{image_size[1]}")

    # --- images: copy each camera subdir ---
    images_dst = dst_dir / "images"
    images_dst.mkdir(parents=True, exist_ok=True)

    cam_dirs = sorted(d for d in images_src.iterdir() if d.is_dir())
    for cam_dir in tqdm(cam_dirs, desc=subject_dir.name, unit="cam"):
        dst_cam = images_dst / cam_dir.name
        if dst_cam.exists():
            shutil.rmtree(dst_cam)
        copy_cam_dir(cam_dir, dst_cam)

    # --- calibration: convert and write ---
    calib_dst = dst_dir / "calibration.json"
    converted = convert_calibration(calib_src, image_size)
    with open(calib_dst, "w") as f:
        json.dump(converted, f, indent=2)
    print(f"  wrote  {calib_dst} ({len(converted)} cameras)")


def main() -> None:
    if not SRC_ROOT.is_dir():
        raise SystemExit(f"Source directory not found: {SRC_ROOT}")

    subjects = sorted(p for p in SRC_ROOT.iterdir() if p.is_dir())
    if not subjects:
        raise SystemExit(f"No subject subdirectories found under {SRC_ROOT}")

    print(f"Found {len(subjects)} subject(s): {[s.name for s in subjects]}")

    for subject_dir in subjects:
        dst_dir = DST_ROOT / subject_dir.name
        print(f"\nProcessing {subject_dir.name} -> {dst_dir}")
        process_subject(subject_dir, dst_dir)

    print(f"\nDone. Run inference with:")
    for subject_dir in subjects:
        print(
            f"  python -m inference run "
            f"--cfg configs/examples/presets/quick.yaml "
            f"--footage {DST_ROOT} "
            f"--seq_name {subject_dir.name} "
            f"--calib {DST_ROOT / subject_dir.name / 'calibration.json'} "
            f"--out-tag thuman4 -v"
        )


if __name__ == "__main__":
    main()
