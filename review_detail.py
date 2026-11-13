"""
review_detail.py - bigger, tighter review strips for chosen candidates (two rows of 7 frames,
-2.4 s .. +2.8 s every 0.4 s), 4 candidates per sheet, so landing / stepping off and the real
start and end of each trick can be judged.

  python review_detail.py <folder with .360> <review folder with candidates.json> <out folder> 0,3,4,...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoreframe as ar  # noqa: E402
import tricks as tk  # noqa: E402

OFFSETS = [round(-2.4 + 0.4 * i, 1) for i in range(14)]
TW, TH = 264, 200
PER_SHEET = 4


def main():
    folder, review, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    wanted = [int(x) for x in sys.argv[4].split(",")]
    global OFFSETS
    if len(sys.argv) > 5:
        a, b, st = (float(x) for x in sys.argv[5].split(":"))
        OFFSETS = [round(a + st * i, 2) for i in range(int(round((b - a) / st)) + 1)]
    nrow = (len(OFFSETS) + 6) // 7
    per_sheet = max(1, 8 // nrow)
    out.mkdir(parents=True, exist_ok=True)
    items = {it["n"]: it for it in json.load(open(review / "candidates.json"))}
    blocks, cur = [], None
    for n in wanted:
        it = items[n]
        if cur is None or cur[0] != it["clip"]:
            p = folder / it["clip"]
            src = ar.open_source(p)
            cur = (it["clip"], src, tk.load_scan(folder / ".autoreframe" / f"{p.stem}.scan.npz"), ar.GpuView(src, TW, TH))
        _, src, s, gv = cur
        times, (yaw, pitch, h) = tk.follow_path(s, it["det"], it["t"] + OFFSETS[0] - 0.6, it["t"] + OFFSETS[-1] + 0.6)
        tiles = []
        for dt in OFFSETS:
            t = min(max(0.0, it["t"] + dt), src.info["duration"] - 0.1)
            gen = src.raw_frames(t, 0.05)
            raw = next(gen, None)
            gen.close()
            if raw is None:
                tiles.append(np.zeros((TH, TW, 3), np.uint8))
                continue
            hh = float(np.interp(t, times, h))
            vfov = float(np.clip(hh / 0.62, 25, 70))
            hfov = float(np.degrees(2 * np.arctan(np.tan(np.radians(vfov) / 2) * TW / TH)))
            v = np.ascontiguousarray(gv.view(raw, hfov, float(np.interp(t, times, yaw)),
                                             float(np.interp(t, times, pitch)) - 4))
            cv2.putText(v, f"{dt:+.1f}", (4, 18), 0, 0.55, (0, 0, 0), 3)
            cv2.putText(v, f"{dt:+.1f}", (4, 18), 0, 0.55, (0, 255, 255), 1)
            tiles.append(v)
        while len(tiles) % 7:
            tiles.append(np.zeros((TH, TW, 3), np.uint8))
        lab = np.zeros((TH * nrow, 70, 3), np.uint8)
        cv2.putText(lab, f"#{n}", (2, 110), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(lab, it["clip"][6:8], (4, 140), 0, 0.5, (180, 180, 180), 1)
        row = np.vstack([np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)])
        blocks.append(np.hstack([lab, row]))
        blocks.append(np.full((6, blocks[-1].shape[1], 3), 255, np.uint8))
        print(f"#{n} done", flush=True)
    per = per_sheet * 2
    for k in range(0, len(blocks), per):
        cv2.imwrite(str(out / f"detail_{k // per:02d}.jpg"), np.vstack(blocks[k:k + per]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("sheets:", (len(blocks) + per - 1) // per)


if __name__ == "__main__":
    main()
