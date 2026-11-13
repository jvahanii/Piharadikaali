"""
trim_ends.py - shorten the ride-away tail of rendered clips the way the user trims by hand.

  python trim_ends.py <folder with .360> <review folder> <rendered clip folder> [--dry-run]

Rule (fitted to the user's LosslessCut trims of 46/part 2, see end_rule.json): end the clip once the
rider's tracked height drops below `frac` of their size at the trick (and below `floor` px in the
scan), but never sooner than `after_peak` s after the speed peak and never later than the judged end.
Long clips (grinds/slides longer than 7 s) and falls keep their judged end, since the rider can shrink
mid-trick there. Trimming copies the stream (no re-encode), so the clips can still be joined with -c copy.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tricks as tk  # noqa: E402

RULE = json.load(open(Path(__file__).resolve().parent / "end_rule.json"))


def rule_end(start, end, peak, times, h):
    g = np.arange(max(start, peak - 1), end, 0.1)
    hh = np.convolve(np.interp(g, times, h), np.ones(3) / 3, "same")
    i0 = int(np.argmax(hh))
    thr = max(RULE["frac"] * hh[i0], RULE["floor"])
    lo = max(i0, int(round((peak + RULE["after_peak"] - g[0]) / 0.1)))
    below = np.where(hh[lo:] < thr)[0]
    e = g[lo + below[0]] + RULE["pad"] if len(below) else end
    return float(min(max(e, peak + RULE["after_peak"]), end))


def main():
    folder, review, clips = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dry = "--dry-run" in sys.argv
    cands = {c["n"]: c for c in json.load(open(review / "candidates.json"))}
    jobs = sorted(json.load(open(review / "jobs.json")), key=lambda j: (cands[j["n"]]["clip"], cands[j["n"]]["t"]))
    prev_end, scans, saved = {}, {}, 0.0
    for j in jobs:
        c = cands[j["n"]]
        stem = Path(c["clip"]).stem
        start = max(0.0, j["start"], prev_end.get(c["clip"], 0.0))
        prev_end[c["clip"]] = j["end"]
        f = clips / f"{stem}_{start:05.1f}s_{j['label']}.mp4"
        if not f.exists():
            print(f"missing {f.name}")
            continue
        if j["label"] == "fall" or j["end"] - start > 7.0:
            print(f"{f.name:32s} kept (fall or long trick)")
            continue
        if stem not in scans:
            scans[stem] = tk.load_scan(folder / ".autoreframe" / f"{stem}.scan.npz")
        times, (_, _, h) = tk.follow_path(scans[stem], c["det"], start - 1, j["end"] + 1)
        new_end = rule_end(start, j["end"], c["t"], times, h)
        cut = j["end"] - new_end
        print(f"{f.name:32s} {j['end'] - start:4.1f}s -> {new_end - start:4.1f}s")
        if cut < 0.2 or dry:
            continue
        tmp = f.with_suffix(".tmp.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-t", f"{new_end - start:.3f}", "-c", "copy",
                        "-map", "0", str(tmp)], check=True)
        tmp.replace(f)
        saved += cut
    print(f"trimmed {saved:.1f} s in total")


if __name__ == "__main__":
    main()
