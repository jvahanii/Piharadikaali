"""
render_selected.py - render chosen candidates from review_strips.py.

  python render_selected.py <folder with .360> <review folder> <out folder> landed=1,4,7 fall=3,9

Landed tricks get 1.9 s before / 2.3 s after the peak; falls get 1.9 s / 3.0 s so the slam is shown.
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoreframe as ar  # noqa: E402
import tricks as tk  # noqa: E402


def main():
    folder, review, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    items = {it["n"]: it for it in json.load(open(review / "candidates.json"))}
    jobs, spans = [], {}
    for arg in sys.argv[4:]:
        if arg.endswith(".json"):  # explicit [{n, start, end, label}] in source seconds
            for j in json.load(open(arg)):
                jobs.append((j["n"], j["label"]))
                spans[j["n"]] = (j["start"], j["end"])
            continue
        label, nums = arg.split("=")
        jobs += [(int(x), label) for x in nums.split(",") if x.strip()]
    jobs.sort(key=lambda j: (items[j[0]]["clip"], items[j[0]]["t"]))
    args = types.SimpleNamespace(fov="auto", min_fov=60, max_fov=110, tilt=-3, smooth=0.35,
                                 width=1920, height=1080, codec=ar.pick_codec("auto"))
    cache, prev_end = {}, {}
    for k, (n, label) in enumerate(jobs):
        it = items[n]
        if it["clip"] not in cache:
            cache.clear()
            p = folder / it["clip"]
            cache[it["clip"]] = (ar.open_source(p), tk.load_scan(folder / ".autoreframe" / f"{p.stem}.scan.npz"))
        src, s = cache[it["clip"]]
        fps = src.info["fps"]
        before, after = 1.9, (3.0 if label == "fall" else 2.3)
        start = max(0.0, it["t"] - before, prev_end.get(it["clip"], 0.0))
        end = min(src.info["duration"] - 0.05, it["t"] + after)
        if n in spans:
            start = max(0.0, spans[n][0], prev_end.get(it["clip"], 0.0))
            end = min(src.info["duration"] - 0.05, spans[n][1])
        prev_end[it["clip"]] = end
        times, (yaw, pitch, h) = tk.follow_path(s, it["det"], start - 0.5, end + 0.5)
        import numpy as np
        vt = start + np.arange(int((end - start) * fps)) / fps
        track = list(zip(np.interp(vt, times, yaw), np.interp(vt, times, pitch), np.interp(vt, times, h)))
        angles = ar.camera_path(track, fps, args)
        name = f"{Path(it['clip']).stem}_{start:05.1f}s_{label}.mp4"
        print(f"[{k + 1}/{len(jobs)}] {name}", flush=True)
        ar.render(src, start, end - start, angles, out / name, args)


if __name__ == "__main__":
    main()
