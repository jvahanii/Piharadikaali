"""
scan.py - one-time scan of a whole 360 clip. Saves every person / skateboard detection, motion
and audio features to <clip folder>/.autoreframe/<clip>.scan.npz so trick finding can be tuned
without decoding the video again.

  python scan.py <.360 files or folders>
"""
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoreframe as ar  # noqa: E402

SCAN_FPS = 10
W, H = 1920, 960
PAD = W // 4


def gpu_frames(src, fps):
    """Decode a GoPro .360 on the GPU at a reduced frame rate -> fixed EAC frames."""
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(src.path),
           "-filter_complex",
           f"[0:v:0]fps={fps},hwdownload,format=nv12[a];[0:v:1]fps={fps},hwdownload,format=nv12[b];"
           "[a][b]vstack,format=bgr24[v]", "-map", "[v]", "-f", "rawvideo", "-"]
    for raw in ar.pipe_frames(cmd, src.raw_w, 2 * src.face):
        yield src._fix(raw)


def cpu_frames(src, fps):
    for i, f in enumerate(src.frames(0, src.info["duration"])):
        if i % max(1, round(src.info["fps"] / fps)) == 0:
            yield f


def audio_flux(path, cache_wav):
    """Onset strength of 1-8 kHz sound per 0.1 s (pops, landings and grinds are sharp sounds)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                    str(cache_wav)], check=True)
    w = wave.open(str(cache_wav))
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    cache_wav.unlink()
    hop, n = 160, 512
    frames = np.lib.stride_tricks.sliding_window_view(a, n)[::hop] * np.hanning(n).astype(np.float32)
    out = []
    for k in range(0, len(frames), 4000):  # chunks to keep memory low
        sp = np.log1p(np.abs(np.fft.rfft(frames[k:k + 4001], axis=1))[:, 32:256] * 10)
        out.append(np.maximum(sp[1:] - sp[:-1], 0).sum(1))
    f = np.concatenate([[0]] + out)
    per = int(16000 / hop / SCAN_FPS)  # 10 ms hops -> 0.1 s bins: keep the max
    f = f[: len(f) // per * per].reshape(-1, per).max(1)
    return f.astype(np.float32)


def colors(frame, x1, y1, x2, y2):
    """Median Lab colour of the upper and lower half of a person box (to recognise you)."""
    x1, x2 = int(max(0, x1)), int(min(frame.shape[1], x2))
    y1, y2 = int(max(0, y1)), int(min(frame.shape[0], y2))
    if x2 - x1 < 2 or y2 - y1 < 4:
        return [np.nan] * 6
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).reshape(-1, (x2 - x1), 3)
    h = crop.shape[0]
    up = crop[int(h * 0.15):int(h * 0.5)].reshape(-1, 3)
    lo = crop[int(h * 0.55):int(h * 0.9)].reshape(-1, 3)
    if len(up) == 0 or len(lo) == 0:
        return [np.nan] * 6
    return list(np.median(up, 0)) + list(np.median(lo, 0))


def scan(path, model):
    t0 = time.time()
    src = ar.open_source(path)
    out = path.parent / ".autoreframe" / f"{path.stem}.scan.npz"
    if out.exists():
        print(f"{path.name}: already scanned")
        return
    out.parent.mkdir(exist_ok=True)
    mx, my = src.maps(*ar.equirect_rays(W, H))
    bg = cv2.createBackgroundSubtractorMOG2(history=SCAN_FPS * 3, varThreshold=32, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dets, share = [], []
    batch, idx = [], []

    def flush():
        res = model.predict([b[0] for b in batch], classes=[0, 36], conf=0.2, imgsz=1920, verbose=False)
        for (fpad, mpad), r, fi in zip(batch, res, idx):
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                             r.boxes.cls.cpu().numpy()):
                if (x1 + x2) / 2 >= W:  # duplicate from the wrap-around padding
                    continue
                roi = mpad[int(y1) // 2:int(y2) // 2, int(x1) // 2:int(x2) // 2]
                moving = float((roi > 0).mean()) if roi.size else 0.0
                col = colors(fpad, x1, y1, x2, y2) if k == 0 else [np.nan] * 6
                dets.append([fi, k, x1, y1, x2, y2, c, moving] + col)
        batch.clear()
        idx.clear()

    try:
        frames = gpu_frames(src, SCAN_FPS)
        first = next(frames)
    except Exception:
        frames, first = cpu_frames(src, SCAN_FPS), None
    import itertools
    stream = itertools.chain([first], frames) if first is not None else frames
    n = 0
    for n, raw in enumerate(stream):
        eq = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
        small = cv2.resize(eq, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        m = cv2.morphologyEx(bg.apply(small), cv2.MORPH_OPEN, kernel)
        share.append(cv2.countNonZero(m) / m.size)
        m = cv2.dilate(m, kernel, iterations=2)
        batch.append((np.concatenate([eq, eq[:, :PAD]], 1), np.concatenate([m, m[:, :PAD // 2]], 1)))
        idx.append(n)
        if len(batch) == 8:
            flush()
        if n % 100 == 0:
            done = n / SCAN_FPS / src.info["duration"] * 100
            print(f"\r{path.name}: {done:5.1f}%", end="", flush=True)
    if batch:
        flush()
    flux = audio_flux(path, out.with_suffix(".wav"))
    np.savez_compressed(out, dets=np.array(dets, np.float32).reshape(-1, 14), share=np.array(share, np.float32),
                        audio=flux, fps=SCAN_FPS, duration=src.info["duration"], W=W, H=H)
    print(f"\r{path.name}: {n + 1} frames, {len(dets)} detections ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ar.need("ffmpeg")
    files = []
    for p in map(Path, sys.argv[1:]):
        files += sorted(p.glob("*.360")) if p.is_dir() else [p]
    model = ar.load_yolo("yolo11s.pt")
    for f in files:
        try:
            scan(f, model)
        except Exception as e:
            print(f"{f.name}: failed: {e}")
