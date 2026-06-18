#!/usr/bin/env python3
"""Flatten <subject>/images/<cam>/ -> <subject>/<cam>/ for all processed subjects."""
import shutil
from pathlib import Path

PROCESSED_ROOT = Path("anchor_data/processed")


def main() -> None:
    if not PROCESSED_ROOT.is_dir():
        raise SystemExit(f"Not found: {PROCESSED_ROOT}")

    for subject_dir in sorted(p for p in PROCESSED_ROOT.iterdir() if p.is_dir()):
        images_dir = subject_dir / "images"
        if not images_dir.is_dir():
            print(f"[skip] no images/ in {subject_dir}")
            continue

        print(f"Flattening {subject_dir}")
        for cam_dir in sorted(d for d in images_dir.iterdir() if d.is_dir()):
            target = subject_dir / cam_dir.name
            if target.exists():
                print(f"  [skip] {target} already exists")
                continue
            shutil.move(str(cam_dir), str(target))

        try:
            images_dir.rmdir()
            print(f"  removed empty {images_dir}")
        except OSError:
            print(f"  [warn] {images_dir} not empty, left in place")

    print("Done.")


if __name__ == "__main__":
    main()