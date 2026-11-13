"""
review_strips.py - make picture strips of every likely trick so each one can be judged
(landed / fall / simple fail) before rendering.

  python review_strips.py <folder with .360 files> <output folder> [--threshold 0.3]

Writes sheet_XX.jpg (8 tricks per sheet, 7 frames each from 1.2 s before to 2.4 s after the
peak) and candidates.json (index -> clip, time, score, detection) for autoedit-style rendering.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoreframe as ar  # noqa: E402
import tricks as tk  # noqa: E402

OFFSETS = [-1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4]
TW, TH = 256, 144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--gap", type=float, default=1.5,
                    help="minimum seconds between candidates; small so back-to-back tricks stay separate")
    args = ap.parse_args()
    folder, out = Path(args.folder), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = tk.load_model()
    items, rows = [], []
    for path in sorted(folder.glob("*.360")):
        scan_file = folder / ".autoreframe" / f"{path.stem}.scan.npz"
        if not scan_file.exists():
            continue
        s = tk.load_scan(scan_file)
        cands = tk.find_candidates(s)
        if not cands:
            continue
        scores = tk.score(model, [c["feats"] for c in cands])
        scores[tk.handling_mask(s, cands)] = 0
        keep = tk.select(cands, scores, args.threshold, gap=args.gap)
        if not keep:
            continue
        src = ar.open_source(path)
        gv = ar.GpuView(src, TW, TH)
        for i in keep:
            c = cands[i]
            times, (yaw, pitch, h) = tk.follow_path(s, c["det"], c["t"] - 2, c["t"] + 3)
            tiles = []
            for dt in OFFSETS:
                t = min(max(0.0, c["t"] + dt), src.info["duration"] - 0.1)
                raw = next(src.raw_frames(t, 0.05), None)
                if raw is None:
                    tiles.append(np.zeros((TH, TW, 3), np.uint8))
                    continue
                hh = float(np.interp(t, times, h))
                fov = float(np.clip(hh / 0.45 * 16 / 9, 60, 110))
                v = gv.view(raw, fov, float(np.interp(t, times, yaw)), float(np.interp(t, times, pitch)) - 3)
                v = np.ascontiguousarray(v)
                cv2.putText(v, f"{dt:+.1f}", (4, 16), 0, 0.45, (0, 255, 255), 1)
                tiles.append(v)
            n = len(items)
            lab = np.zeros((TH, 80, 3), np.uint8)
            cv2.putText(lab, f"#{n}", (4, 70), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(lab, path.stem[-2:], (4, 100), 0, 0.5, (180, 180, 180), 1)
            rows.append(np.hstack([lab] + tiles))
            items.append({"n": n, "clip": path.name, "t": c["t"], "score": float(scores[i]), "det": c["det"]})
            print(f"#{n} {path.stem} {c['t']:.1f}s score {scores[i]:.2f}", flush=True)
    for k in range(0, len(rows), 8):
        cv2.imwrite(str(out / f"sheet_{k // 8:02d}.jpg"), np.vstack(rows[k:k + 8]))
    json.dump(items, open(out / "candidates.json", "w"), indent=1)
    print(f"{len(items)} candidates, {(len(rows) + 7) // 8} sheets")


if __name__ == "__main__":
    main()
