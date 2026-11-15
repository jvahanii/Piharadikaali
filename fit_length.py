"""
fit_length.py - fill a compilation to an exact length with fail clips, then join it.

  python fit_length.py <clip folder> <seconds> <output.mp4> [extra fail folders ...]

Landed and fall clips (`*_landed.mp4`, `*_fall.mp4`) in <clip folder> are always kept. Fail
clips (`*_fail.mp4`) from the folder, its `unused_fails` subfolder and any extra folders form
the pool. Fails are dropped evenly across the session until the total is just over the target,
the dropped ones go to `unused_fails` (nothing is deleted), the longest kept fail is shortened
at its end to hit the length exactly, and everything is joined in session order with one
re-encode (a plain -c copy join glitches at seams after stream-copy trims).
"""
import shutil
import subprocess
import sys
from pathlib import Path


def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def main():
    folder, target, output = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    spare = folder / "unused_fails"
    spare.mkdir(exist_ok=True)
    for src in [spare] + [Path(p) for p in sys.argv[4:]]:
        for f in src.glob("*_fail.mp4"):
            if src != folder:
                shutil.move(str(f), folder / f.name)
    dur = {f: duration(f) for f in folder.glob("*.mp4")}
    fixed = sum(d for f, d in dur.items() if not f.name.endswith("_fail.mp4"))
    fails = sorted(f for f in dur if f.name.endswith("_fail.mp4"))
    need = target - fixed
    if need < 0:
        sys.exit(f"landed + falls already run {fixed:.1f} s, over the {target:.1f} s target - nothing done")
    if sum(dur[f] for f in fails) < need:
        sys.exit(f"not enough fails: need {need:.1f} s, have {sum(dur[f] for f in fails):.1f} s")
    keep = list(fails)
    while len(keep) > 1 and sum(dur[f] for f in keep) - dur[min(keep, key=lambda f: dur[f])] >= need:
        # drop from evenly spread positions so no part of the session loses all its fails
        dropped = len(fails) - len(keep)
        keep.pop(int((dropped * 0.618 % 1) * len(keep)))
    for f in fails:
        if f not in keep:
            shutil.move(str(f), spare / f.name)
    excess = fixed + sum(dur[f] for f in keep) - target
    if excess > 0.02:
        victim = max(keep, key=lambda f: dur[f])
        tmp = victim.with_suffix(".tmp.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(victim), "-t", f"{dur[victim] - excess:.3f}",
                        "-map", "0", "-c", "copy", str(tmp)], check=True)
        tmp.replace(victim)
        print(f"shortened {victim.name} by {excess:.2f} s")
    clips = sorted(folder.glob("*.mp4"))
    (folder / "list.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(folder / "list.txt"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{target:.3f}", "-movflags", "+faststart", str(output)],
                   check=True)
    n_fail = sum(c.name.endswith("_fail.mp4") for c in clips)
    print(f"{len(clips)} clips ({n_fail} fails, {len(fails) - len(keep)} set aside) -> {output.name}, "
          f"{duration(output):.2f} s")


if __name__ == "__main__":
    main()
