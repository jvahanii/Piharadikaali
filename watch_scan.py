"""
watch_scan.py - scan .360 files while they are still being copied off the camera.

  python watch_scan.py <folder>

A file is scanned once its size has not changed for STABLE seconds and ffprobe can read it.
Exits after QUIET seconds with no new or growing files once everything is scanned.
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import autoreframe as ar  # noqa: E402
import scan as sc  # noqa: E402

STABLE, QUIET = 45, 20 * 60


def readable(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() not in ("", "N/A")


def main():
    folder = Path(sys.argv[1])
    sizes, stable_since, last_change = {}, {}, time.time()
    model = ar.load_yolo(str(REPO / "yolo11s.pt"))
    while True:
        now = time.time()
        pending = 0
        for p in sorted(folder.glob("*.360")):
            if (folder / ".autoreframe" / f"{p.stem}.scan.npz").exists():
                continue
            pending += 1
            size = p.stat().st_size
            if sizes.get(p) != size:
                sizes[p], stable_since[p], last_change = size, now, now
                continue
            if now - stable_since[p] < STABLE or not readable(p):
                continue
            print(f"scanning {p.name} ({size / 1e9:.1f} GB)", flush=True)
            try:
                sc.scan(p, model)
            except Exception as e:  # keep going with the other clips
                print(f"{p.name}: failed: {e}", flush=True)
                sizes[p] = -1
            last_change = time.time()
        if pending == 0 and time.time() - last_change > QUIET:
            print("all scanned, no new files for 20 min", flush=True)
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
