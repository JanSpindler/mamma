#!/usr/bin/env python3
"""Convert a VR-eyetracking dome capture to MAMMA-compatible layout.

The dome capture is *frame-major*; MAMMA needs *camera-major*.

Source:  <src>/
           frame_00000/image/color_C0000.jpg
           frame_00000/mask/mask_C0000.jpg      (unused, see NOTES)
           ...
           frame_01047/image/color_C1005.jpg
           calibration_dome.json                (dome format v3.0.0)
           background/, preview/, timestamps.csv, ...   (ignored)

Output:  <dst>/<seq_name>/
           C0000/frame_00000.jpg                (copied, byte-identical)
           C0000/frame_00001.jpg
           ...
           C1005/frame_01047.jpg
           calibration.json                     (MAMMA OpenCV-flat format)

The source tree is opened read-only and is never modified.

Run:
    python scripts/convert_vr_dome.py --dry-run             # counts + disk check
    python scripts/convert_vr_dome.py --frame-end 120       # short slice
    python scripts/convert_vr_dome.py                       # everything

Then:
    python -m inference run \\
        --cfg      configs/examples/presets/vr_dome.yaml \\
        --footage  data/vr_dome \\
        --seq_name 2026_07_27_VR_eyetracking \\
        --calib    data/vr_dome/2026_07_27_VR_eyetracking/calibration.json \\
        --out-tag  vr_dome -v

NOTES
  * No ``images/`` wrapper. ``capture.discovery.find_image_cam_dirs`` accepts
    only directories one level under the sequence dir that *directly* contain
    image files, so the camera dirs must sit at the top of <seq_name>/.
    ``calibration.json`` living beside them is fine -- discovery filters on
    ``os.path.isdir``.
  * Camera dir names are the calibration ids verbatim (C0000, ...).
    ``normalize_cam_name`` only strips whitespace and a trailing .npz/.mp4,
    and ``run_ma_cap._select_camera`` needs an exact key match.
  * Frame filenames keep the source frame-dir name, so a plain lexicographic
    sort (what ``run_ma_cap._gather_images`` does) is frame order, and the
    mapping back to the source stays visible in ``ls``. Extensions stay
    lowercase: that function's IMG_EXTENSIONS is lowercase-only.
  * With --frame-start 0 --frame-stride 1 the pipeline's positional frame
    index equals the source frame number, so ``global.start_frame: 500``
    means frame_00500. Any other slicing breaks that identity.
  * The capture's own per-frame masks are NOT converted. They are JPEGs with
    compression ringing (hundreds of connected components at threshold 0),
    and ``run_ma_2d`` derives the person bbox from *any* nonzero pixel --
    reusing them naively yields a full-image bbox. Let ma_masks run SAM+YOLO.
  * ``background/`` holds 32 subject-free plates. MAMMA has no
    background-subtraction path today; kept here only as a pointer.
  * The capture runs at 25 fps but ``run_ma_cap._ingest_images_root_mode``
    hardcodes 30 in images mode. Pass ``--fps 25`` via the preset's
    ``ma_cap.flags``; see configs/examples/presets/vr_dome.yaml.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

SRC_DEFAULT = Path("/data/vci/VR_eyetracking/2026_07_27_VR_eyetracking")
DST_DEFAULT = Path("data/vr_dome")
CALIB_NAME = "calibration_dome.json"

FRAME_DIR_RE = re.compile(r"^frame_(\d+)$")
IMAGE_NAME_RE = re.compile(r"^color_(.+)\.jpg$")

# Refuse to fill the volume completely; /data is chronically near-full.
MIN_FREE_BYTES = 50 * 1024**3


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def discover_frames(src: Path, start: int, end, stride: int) -> list[Path]:
    """Sorted frame directories, sliced by [start, end) with the given stride."""
    frames = []
    for p in sorted(src.iterdir()):
        m = FRAME_DIR_RE.match(p.name)
        if m and p.is_dir():
            frames.append((int(m.group(1)), p))
    if not frames:
        raise SystemExit(f"No frame_* directories found under {src}")

    numbers = [n for n, _ in frames]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        print(f"  [warn] frame numbering has gaps ({numbers[0]}..{numbers[-1]}, "
              f"{len(numbers)} dirs); positional indices will not match names")

    paths = [p for _, p in frames]
    return paths[start:end:stride]


def discover_cams(calib_ids: list[str], probe_frame: Path, wanted) -> list[str]:
    """Cross-check calibration ids against the images, then apply --cams."""
    image_dir = probe_frame / "image"
    if not image_dir.is_dir():
        raise SystemExit(f"No image/ dir in {probe_frame}")

    found = set()
    for f in image_dir.iterdir():
        m = IMAGE_NAME_RE.match(f.name)
        if m:
            found.add(m.group(1))

    calib_set = set(calib_ids)
    if found != calib_set:
        raise SystemExit(
            f"Camera mismatch between {image_dir} and the calibration:\n"
            f"  only in images:      {sorted(found - calib_set)}\n"
            f"  only in calibration: {sorted(calib_set - found)}"
        )

    cams = sorted(calib_set)
    if wanted:
        missing = set(wanted) - calib_set
        if missing:
            raise SystemExit(f"Unknown camera(s): {sorted(missing)}")
        cams = [c for c in cams if c in set(wanted)]
    return cams


def convert_calibration(src_json: Path, cams: list[str], probe_frame: Path,
                        verify_size: bool) -> dict:
    """Convert dome calibration_dome.json to MAMMA OpenCV-flat format.

    Only the top-level ``cameras`` block is read. ``processing_applied`` and
    ``raw_calibration`` are deliberately ignored: the top-level intrinsics
    already equal the final entry of the debayer->superkernel->rotate chain
    for every camera, so they describe the delivered JPEGs as-is.
    """
    with open(src_json) as f:
        src = json.load(f)

    by_id = {c["camera_id"]: c for c in src["cameras"]}
    out = {}
    for cam_name in cams:
        c = by_id[cam_name]

        # 4x4 row-major, world->cam, OpenCV axes -- the same convention
        # capture/loaders/json_loader.py:_parse_opencv_flat assumes, so the
        # top 3 rows drop straight in with no inversion or sign flip.
        V = np.asarray(c["extrinsics"]["view_matrix"], dtype=np.float64).reshape(4, 4)
        R = V[:3, :3]
        t = V[:3, 3]
        if not np.allclose(V[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise SystemExit(f"{cam_name}: view_matrix bottom row is {V[3]}, expected [0,0,0,1]")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise SystemExit(f"{cam_name}: view_matrix rotation block is not orthonormal")
        # optimization/utils/fitting.py rescales when max|t| > 200, assuming mm.
        if not 0.1 < np.abs(t).max() < 100.0:
            raise SystemExit(
                f"{cam_name}: translation {t} does not look like metres; "
                f"downstream code auto-converts anything over 200 as mm"
            )

        K = np.asarray(c["intrinsics"]["camera_matrix"], dtype=np.float64).reshape(3, 3)

        dist = list(c["intrinsics"]["distortion_coefficients"])
        if len(dist) != 5:
            raise SystemExit(f"{cam_name}: expected 5 distortion coefficients, got {len(dist)}")
        if any(abs(float(x)) > 1e-9 for x in dist):
            # MAMMA's --undistort is a no-op for opencv_brown calibrations
            # (capture/undistort.py:_is_noop), so nonzero coefficients would be
            # silently dropped rather than applied.
            raise SystemExit(
                f"{cam_name}: nonzero distortion {dist}; the images were expected to be "
                f"pre-undistorted and MAMMA will not undistort an opencv_brown calibration"
            )

        width, height = (int(round(float(v))) for v in c["intrinsics"]["resolution"])
        if verify_size:
            # run_ma_cap._write_cam_npz takes cam_img_w/h from the calibration and
            # never looks at the JPEGs, so a disagreement here propagates silently
            # through the whole pipeline.
            img_path = probe_frame / "image" / f"color_{cam_name}.jpg"
            with PILImage.open(img_path) as img:
                actual = img.size
            if actual != (width, height):
                raise SystemExit(
                    f"{cam_name}: calibration says {width}x{height} but "
                    f"{img_path} is {actual[0]}x{actual[1]}"
                )

        out[cam_name] = {
            "intrinsic_matrix": K.tolist(),
            "extrinsics_matrix": V[:3, :].tolist(),
            "distortions": [float(x) for x in dist],
            "image_size": [width, height],
        }
    return out


def plan_copies(frames: list[Path], cams: list[str], dst_seq: Path) -> tuple[list, int]:
    """Build the (src, dst) work list and total the bytes still to be written."""
    jobs = []
    todo_bytes = 0
    for cam in tqdm(cams, desc="scanning", unit="cam", leave=False):
        cam_dst = dst_seq / cam
        for frame_dir in frames:
            src_file = frame_dir / "image" / f"color_{cam}.jpg"
            if not src_file.is_file():
                raise SystemExit(f"Missing source image: {src_file}")
            jobs.append((src_file, cam_dst / f"{frame_dir.name}.jpg"))
            todo_bytes += src_file.stat().st_size
    return jobs, todo_bytes


def copy_images(jobs: list, force: bool) -> dict:
    """Copy each image, skipping ones already present at the right size."""
    counts = {"copied": 0, "kept": 0, "replaced": 0}
    for src_file, dst_file in tqdm(jobs, desc="copying", unit="img"):
        if dst_file.exists():
            if not force and dst_file.stat().st_size == src_file.stat().st_size:
                counts["kept"] += 1
                continue
            counts["replaced"] += 1
        else:
            counts["copied"] += 1
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, default=SRC_DEFAULT,
                        help=f"Frame-major capture root (default: {SRC_DEFAULT})")
    parser.add_argument("--dst", type=Path, default=DST_DEFAULT,
                        help=f"Footage root to write into (default: {DST_DEFAULT})")
    parser.add_argument("--seq-name", default=None,
                        help="Sequence dir name under --dst (default: basename of --src)")
    parser.add_argument("--calib-name", default=CALIB_NAME,
                        help=f"Calibration filename inside --src (default: {CALIB_NAME})")
    parser.add_argument("--frame-start", type=int, default=0,
                        help="First source frame index, inclusive (default: 0)")
    parser.add_argument("--frame-end", type=int, default=None,
                        help="Last source frame index, exclusive (default: all)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth frame (default: 1)")
    parser.add_argument("--cams", default=None,
                        help="Comma-separated camera subset, e.g. C0000,C0013")
    parser.add_argument("--force", action="store_true",
                        help="Re-copy images that already exist and overwrite calibration.json")
    parser.add_argument("--no-verify-size", action="store_true",
                        help="Skip the calibration-vs-JPEG image size cross-check")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts and disk usage, write nothing")
    args = parser.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    calib_src = src / args.calib_name
    if not calib_src.is_file():
        raise SystemExit(f"Calibration not found: {calib_src}")

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    seq_name = args.seq_name or src.name
    dst_seq = args.dst / seq_name

    frames = discover_frames(src, args.frame_start, args.frame_end, args.frame_stride)
    if not frames:
        raise SystemExit("Frame range selected no frames")

    with open(calib_src) as f:
        calib_ids = [c["camera_id"] for c in json.load(f)["cameras"]]
    wanted = [c.strip() for c in args.cams.split(",")] if args.cams else None
    cams = discover_cams(calib_ids, frames[0], wanted)

    print(f"Source      {src}")
    print(f"Destination {dst_seq}")
    print(f"Frames      {len(frames)} ({frames[0].name} .. {frames[-1].name}, "
          f"stride {args.frame_stride})")
    print(f"Cameras     {len(cams)} ({cams[0]} .. {cams[-1]})")

    jobs, todo_bytes = plan_copies(frames, cams, dst_seq)
    free = shutil.disk_usage(args.dst if args.dst.exists() else Path.cwd()).free
    print(f"Images      {len(jobs)}")
    print(f"Size        {human(todo_bytes)} to copy, {human(free)} free")

    if args.dry_run:
        print("\nDry run -- nothing written.")
        return

    if free - todo_bytes < MIN_FREE_BYTES:
        raise SystemExit(
            f"Refusing to copy: would leave {human(free - todo_bytes)} free, "
            f"under the {human(MIN_FREE_BYTES)} floor. Narrow the range with "
            f"--frame-end / --frame-stride / --cams."
        )

    calibration = convert_calibration(calib_src, cams, frames[0],
                                      verify_size=not args.no_verify_size)

    dst_seq.mkdir(parents=True, exist_ok=True)
    counts = copy_images(jobs, args.force)
    print(f"  copied {counts['copied']}, kept {counts['kept']}, "
          f"replaced {counts['replaced']}")

    calib_dst = dst_seq / "calibration.json"
    if calib_dst.exists() and not args.force:
        print(f"  [skip] {calib_dst} already exists (use --force to overwrite)")
    else:
        with open(calib_dst, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"  wrote  {calib_dst} ({len(calibration)} cameras)")

    print("\nDone. Run inference with:")
    print(
        f"  python -m inference run"
        f" --cfg configs/examples/presets/vr_dome.yaml"
        f" --footage {args.dst}"
        f" --seq_name {seq_name}"
        f" --calib {calib_dst}"
        f" --out-tag vr_dome -v"
    )


if __name__ == "__main__":
    main()
