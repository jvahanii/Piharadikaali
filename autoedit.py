"""
autoedit.py - find every trick in your GoPro MAX .360 clips and cut each one into a short 16:9 clip.

  python autoedit.py <.360 files or folders> [options]

For each clip:
  1. scan the whole clip once (people, skateboards, motion, sound; cached in .autoreframe/)
  2. find moments where you're on your board at peak speed and score how trick-like they are
     (model.json, learned from your own edits - see train_model.py)
  3. cut ~4 s around every likely trick and render it, following you, into 'tricks/'

Output names: <clip>_<time>s_<score>.mp4 - higher score = more likely a trick you'd keep.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoreframe as ar  # noqa: E402
import scan  # noqa: E402
import tricks as tk  # noqa: E402


def render_trick(src, s, cand, sc, args, out_dir):
    fps = src.info["fps"]
    start = max(0.0, cand["t"] - args.before)
    end = min(src.info["duration"] - 0.05, cand["t"] + args.after)
    times, (yaw, pitch, h) = tk.follow_path(s, cand["det"], start - 0.5, end + 0.5)
    n = int((end - start) * fps)
    vt = start + np.arange(n) / fps
    track = list(zip(np.interp(vt, times, yaw), np.interp(vt, times, pitch), np.interp(vt, times, h)))
    angles = ar.camera_path(track, fps, args)
    name = f"{src.path.stem}_{cand['t']:05.1f}s_{int(round(sc * 100)):02d}.mp4"
    ar.render(src, start, end - start, angles, out_dir / name, args)
    return name


def process(path, args, yolo, model):
    t0 = time.time()
    scan_file = path.parent / ".autoreframe" / f"{path.stem}.scan.npz"
    if not scan_file.exists():
        scan.scan(path, yolo())
    s = tk.load_scan(scan_file)
    cands = tk.find_candidates(s)
    if not cands:
        print("  no tricks found")
        return
    scores = tk.score(model, [c["feats"] for c in cands])
    scores[tk.handling_mask(s, cands)] = 0  # you're touching the camera, not doing a trick
    keep = tk.select(cands, scores, args.threshold)
    print(f"  {len(keep)} likely tricks (of {len(cands)} moments on the board)")
    if args.list:
        for i in keep:
            print(f"    {cands[i]['t']:7.1f}s  score {scores[i]:.2f}")
        return
    out_dir = Path(args.out) if args.out else path.parent / "tricks"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ar.open_source(path)
    for k, i in enumerate(keep):
        print(f"  [{k + 1}/{len(keep)}] trick at {cands[i]['t']:.1f}s (score {scores[i]:.2f})")
        render_trick(src, s, cands[i], scores[i], args, out_dir)
    print(f"  done in {time.time() - t0:.0f}s -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Find and cut every trick in GoPro MAX .360 clips.")
    ap.add_argument("inputs", nargs="+", help=".360 files or folders")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="how sure it must be: lower = more clips (finds more tricks, more extras) (0.4)")
    ap.add_argument("--before", type=float, default=1.9, help="seconds before the trick (1.9)")
    ap.add_argument("--after", type=float, default=2.3, help="seconds after the trick (2.3)")
    ap.add_argument("--list", action="store_true", help="only list the tricks found, don't render")
    ap.add_argument("--fov", default="auto", help="'auto' (zoom to you) or horizontal degrees, e.g. 90")
    ap.add_argument("--min-fov", type=float, default=60)
    ap.add_argument("--max-fov", type=float, default=110)
    ap.add_argument("--tilt", type=float, default=-3)
    ap.add_argument("--smooth", type=float, default=0.35, help="camera smoothing in seconds (0.35)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--codec", default="auto")
    ap.add_argument("--out", help="output folder (default: 'tricks' next to the clips)")
    args = ap.parse_args()
    if args.fov != "auto":
        args.fov = float(args.fov)
    ar.need("ffmpeg")
    ar.need("ffprobe")
    args.codec = ar.pick_codec(args.codec)
    if not tk.MODEL.exists():
        sys.exit("model.json is missing - run train_model.py first.")
    model = tk.load_model()

    files = []
    for p in map(Path, args.inputs):
        files += sorted(p.glob("*.360")) if p.is_dir() else [p]
    cache = {}

    def yolo():
        if "m" not in cache:
            cache["m"] = ar.load_yolo("yolo11s.pt")
        return cache["m"]

    for f in files:
        print(f"\n{f.name}")
        try:
            process(f, args, yolo, model)
        except Exception as e:
            print(f"  failed: {e}")


if __name__ == "__main__":
    main()
