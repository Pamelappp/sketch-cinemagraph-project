"""
Debug script: inspect animation frames for duplicate / stutter frames.

Usage:
    python check_animation_frames.py                         # uses path from default.yaml
    python check_animation_frames.py path/to/output.gif
    python check_animation_frames.py path/to/output.mp4
"""

import sys
from pathlib import Path

import numpy as np


def load_frames(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext == ".gif":
        import imageio.v2 as imageio
        reader = imageio.get_reader(path)
        frames = [np.asarray(f) for f in reader]
        reader.close()
    elif ext in (".mp4", ".avi", ".mov"):
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        raise ValueError(f"Unsupported format: {ext}")

    return frames


def frame_mad(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def analyze(path: str):
    print(f"\n=== Animation Frame Analysis: {path} ===\n")
    frames = load_frames(path)
    n = len(frames)
    print(f"Total frames: {n}")

    if n < 2:
        print("[Warning] Only 1 frame — cannot analyse motion.")
        return

    diffs = [frame_mad(frames[i], frames[i + 1]) for i in range(n - 1)]

    min_diff = min(diffs)
    min_idx = diffs.index(min_diff)
    max_diff = max(diffs)
    mean_diff = float(np.mean(diffs))

    print(f"Adjacent-frame MAD  →  min={min_diff:.4f}  max={max_diff:.4f}  mean={mean_diff:.4f}")
    print(f"Minimum-diff pair   →  frames [{min_idx}] and [{min_idx + 1}]  (MAD={min_diff:.4f})")

    for i, d in enumerate(diffs):
        if d == 0.0:
            print(f"[Warning] Frames {i} and {i + 1} are COMPLETELY IDENTICAL (MAD=0).")
        elif d < 0.5:
            print(f"[Notice]  Frames {i} and {i + 1} are nearly identical (MAD={d:.4f}).")

    first_last_mad = frame_mad(frames[0], frames[-1])
    print(f"\nFirst vs last frame MAD: {first_last_mad:.4f}")

    if first_last_mad == 0.0:
        print("[Warning] First frame and last frame are COMPLETELY IDENTICAL.")
        print("          This causes a visible stutter/pause at the loop boundary.")
        print("  → Fix: save frames[:-1] instead of all frames,")
        print("          or use np.linspace(..., endpoint=False) when generating frame timestamps.")
    elif first_last_mad < 1.0:
        print("[Notice]  First and last frames are very similar (MAD < 1). Minor stutter possible.")
    else:
        print("[OK]      First and last frames are distinct — loop boundary should be smooth.")

    print("\nPer-frame MAD:")
    for i, d in enumerate(diffs):
        bar_len = int(d / max(max_diff, 1e-6) * 40)
        bar = "█" * bar_len
        print(f"  [{i:4d}→{i + 1:4d}]  {d:7.3f}  {bar}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        try:
            import yaml
            cfg_path = Path(__file__).parent / "configs" / "default.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            path = cfg.get("output", {}).get("final_video_path", "")
            if not path:
                print("No path given and final_video_path not set in config.")
                sys.exit(1)
            path = str(Path(__file__).parent / path)
        except Exception as e:
            print(f"Could not read config: {e}")
            sys.exit(1)

    analyze(path)


if __name__ == "__main__":
    main()
