"""
train_model.py - teach the trick finder from your own hand-made edits.

  python train_model.py <folder with .360 files> <folder with your edits> [more pairs...]

Your edits are matched to their .360 source by file name prefix (GS016712*.mp4 -> GS016712.360)
and by the GoPro timecode they keep, so the exact time of every edit is known. The .360 files are
scanned first if needed (slow, one time). Writes model.json next to this script.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoreframe as ar  # noqa: E402
import scan  # noqa: E402
import tricks as tk  # noqa: E402


def timecode(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream_tags=timecode:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    d = json.loads(out)
    tcs = [s["tags"]["timecode"] for s in d.get("streams", []) if "timecode" in s.get("tags", {})]
    if not tcs:
        return None, None
    h, m, s_, f = map(int, tcs[0].replace(";", ":").split(":"))
    return h * 3600 + m * 60 + s_ + f / 59.94, float(d["format"]["duration"])


def edits_for(src, edit_dir):
    t0, _ = timecode(src)
    res = []
    for e in sorted(edit_dir.glob(src.stem + "*.mp4")):
        t, dur = timecode(e)
        if t is not None and t0 is not None:
            res.append((t - t0, t - t0 + dur))
    return res


def main():
    ar.need("ffprobe")
    pairs = [(Path(a), Path(b)) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
    data, model_yolo = [], None
    for src_dir, edit_dir in pairs:
        for src in sorted(src_dir.glob("*.360")):
            edits = edits_for(src, edit_dir)
            if not edits:
                continue
            scan_file = src.parent / ".autoreframe" / f"{src.stem}.scan.npz"
            if not scan_file.exists():
                model_yolo = model_yolo or ar.load_yolo("yolo11s.pt")
                scan.scan(src, model_yolo)
            cands = tk.find_candidates(tk.load_scan(scan_file))
            for c in cands:
                c["label"] = any(a - 0.7 <= c["t"] <= b + 0.7 for a, b in edits)
            data.append((src.stem, edits, cands))
            print(f"{src.stem}: {len(edits)} edits, {len(cands)} candidates")
    if not data:
        sys.exit("No .360 files with matching edits found.")

    # honest score: each clip judged by a model trained without it
    X = [c["feats"] for _, _, cs in data for c in cs]
    y = [c["label"] for _, _, cs in data for c in cs]
    clip = [n for n, _, cs in data for _ in cs]
    prob = np.zeros(len(y))
    for name, _, _ in data:
        m = np.array([c == name for c in clip])
        if m.all():
            continue
        mod = tk.fit(np.array(X)[~m], np.array(y)[~m])
        prob[m] = tk.score(mod, np.array(X)[m])
    print("\nthreshold  found / your edits   extra clips")
    for thr in (0.3, 0.35, 0.4, 0.5, 0.6, 0.7):
        found = extra = total = 0
        k = 0
        for name, edits, cands in data:
            p = prob[k:k + len(cands)]
            k += len(cands)
            sel = tk.select(cands, p, thr)
            total += len(edits)
            found += sum(any(a - 0.7 <= cands[i]["t"] <= b + 0.7 for i in sel) for a, b in edits)
            extra += sum(not any(a - 0.7 <= cands[i]["t"] <= b + 0.7 for a, b in edits) for i in sel)
        print(f"  {thr:.2f}      {found:3d} / {total:3d}          {extra}")
    model = tk.fit(np.array(X), np.array(y))
    json.dump(model, open(tk.MODEL, "w"), indent=1)
    print(f"\nsaved {tk.MODEL}")


if __name__ == "__main__":
    main()
