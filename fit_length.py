"""
fit_length.py - fill a compilation to an exact length with fail clips, then join it.

  python fit_length.py <clip folder> <seconds> <output.mp4> [extra fail folders ...]
  python fit_length.py <clip folder> 0 <output.mp4> --only landed,fall     (just join, no filling)

Clips in another format (e.g. your own GoPro Player exports: HEVC, 59.94 fps) are converted to a
temporary matching copy for the join; the files themselves are left as they are.

Landed and fall clips (`*_landed.mp4`, `*_fall.mp4`) in <clip folder> are always kept. Fail
clips (`*_fail.mp4`) from the folder, its `unused_fails` subfolder and any extra folders form
the pool. Fails are dropped evenly across the session until the total is just over the target,
the dropped ones go to `unused_fails` (nothing is deleted), the longest kept fail is shortened
at its end to hit the length exactly, and everything is joined in session order with one
re-encode (a plain -c copy join glitches at seams after stream-copy trims).
"""
import shutil
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def duration(p):
    """Video stream length: what the join actually adds (container length can include longer audio)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[0])


def probe(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,r_frame_rate", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    return out.strip().splitlines()[0] if out.strip() else ""


def prepare(clips, tmpdir):
    """Map each clip to the file to join: itself, or a temporary copy matching the common format."""
    formats = {c: probe(c) for c in clips}
    common = max(set(formats.values()), key=list(formats.values()).count)
    codec, w, h, rate = common.split(",")
    out = {}
    for c, f in formats.items():
        out[c] = c
        if f != common:
            fixed = Path(tmpdir) / c.name
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(c), "-map", "0:v:0", "-map", "0:a:0",
                            "-vf", f"scale={w}:{h},fps={rate},format=yuv420p", "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                            str(fixed)], check=True)
            print(f"converted a copy of {c.name} ({f}) for the join")
            out[c] = fixed
    return out


def joined_length(files, tmpdir):
    """Real length of the joined video, measured with a fast stream-copy join."""
    lst, out = Path(tmpdir) / "measure.txt", Path(tmpdir) / "measure.mp4"
    lst.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files))
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v:0",
                    "-c", "copy", str(out)], check=True)
    length = duration(out)
    out.unlink()
    return length


def join(files, output, target=None):
    """Join files in order with one re-encode."""
    lst = Path(tempfile.mkdtemp()) / "list.txt"
    lst.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v:0",
           "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-write_tmcd", "0", "-movflags", "+faststart"]
    if target:
        cmd += ["-t", f"{target:.3f}"]
    subprocess.run(cmd + [str(output)], check=True)
    shutil.rmtree(lst.parent, ignore_errors=True)


def main():
    args, only = list(sys.argv[1:]), None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        del args[i:i + 2]
    folder, target, output = Path(args[0]), float(args[1]), Path(args[2])
    with tempfile.TemporaryDirectory() as tmpdir:
        if target == 0:
            labels = tuple(f"_{l}.mp4" for l in (only or "landed,fall,fail").split(","))
            clips = sorted(c for c in folder.glob("*.mp4") if c.name.endswith(labels))
            files = prepare(clips, tmpdir)
            join([files[c] for c in clips], output)
            print(f"{len(clips)} clips -> {output.name}, {duration(output):.2f} s")
        else:
            fill(folder, target, output, args[3:], tmpdir)


def fill(folder, target, output, extra, tmpdir):
    spare = folder / "unused_fails"
    spare.mkdir(exist_ok=True)
    for src in [spare] + [Path(p) for p in extra]:
        for f in src.glob("*_fail.mp4"):
            if src != folder:
                shutil.move(str(f), folder / f.name)
    files = prepare(sorted(folder.glob("*.mp4")), tmpdir)
    dur = {f: duration(files[f]) for f in files}  # measured on what is actually joined
    fixed = sum(d for f, d in dur.items() if not f.name.endswith("_fail.mp4"))
    fails = sorted(f for f in dur if f.name.endswith("_fail.mp4"))
    # joins lose a little at each seam, so pick fails for a bit more than needed and measure the real length
    slack = 0.02 * len(dur) + 0.5
    need = target + slack - fixed
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
    clips = sorted(folder.glob("*.mp4"))
    excess = joined_length([files[c] for c in clips], tmpdir) - target
    if excess > 0.02:
        victim = max((f for f in keep if files[f] == f), key=lambda f: dur[f])
        tmp = victim.with_suffix(".tmp.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(victim), "-t", f"{dur[victim] - excess:.3f}",
                        "-map", "0", "-c", "copy", str(tmp)], check=True)
        tmp.replace(victim)
        print(f"shortened {victim.name} by {excess:.2f} s")
    elif excess < -0.05:
        print(f"warning: the joined clips come out {-excess:.2f} s short of the target")
    clips = sorted(folder.glob("*.mp4"))
    join([files[c] for c in clips], output, target)
    n_fail = sum(c.name.endswith("_fail.mp4") for c in clips)
    print(f"{len(clips)} clips ({n_fail} fails, {len(fails) - len(keep)} set aside) -> {output.name}, "
          f"{duration(output):.2f} s")


if __name__ == "__main__":
    main()
