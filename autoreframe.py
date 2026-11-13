"""
autoreframe.py - automatically reframe the end of 360 skate clips into normal 16:9 video.

Built for a stationary 360 camera (e.g. GoPro MAX on a tripod or the ground):
  1. Takes the last N seconds of each clip (where the trick is).
  2. Finds the skater in every frame: AI person detection (YOLO) combined with
     motion detection, so people standing still are ignored.
  3. Smooths the path like a camera operator would and auto-zooms to the skater.
  4. Renders a flat 16:9 video (with audio) that follows the skater.

Input: GoPro MAX .360 files directly, or equirectangular (2:1) .mp4 exports.

Usage:
  python autoreframe.py GS016717.360
  python autoreframe.py "C:\\Videos\\skate\\session"         (every .360/.mp4 in a folder)
  python autoreframe.py clip.360 --seconds 8 --fov 80 --debug
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ANALYSIS_W, ANALYSIS_H = 1920, 960  # equirect size used for detection


# ---------------------------------------------------------------- ffmpeg helpers

def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"Error: '{tool}' was not found. Install ffmpeg and make sure it is on PATH.")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"w": int(s["width"]), "h": int(s["height"]),
            "fps": float(num) / float(den), "duration": float(info["format"]["duration"])}


def pick_codec(requested):
    if requested != "auto":
        return requested
    test = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=1280x720", "-frames:v", "1",
                           "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    return "h264_nvenc" if test.returncode == 0 else "libx264"


def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return bool(out.strip())


# ---------------------------------------------------------------- sources
#
# World coordinates: x = right, y = up, z = forward (the camera's front lens).
# A source turns ray directions (x, y, z) into pixel lookup tables for cv2.remap.

class EquirectSource:
    def __init__(self, path, info):
        self.path, self.info = path, info
        self.w, self.h = info["w"], info["h"]

    def frames(self, start, length):
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(self.path),
               "-t", f"{length:.3f}", "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        yield from pipe_frames(cmd, self.w, self.h)

    def maps(self, x, y, z):
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -1, 1))
        mx = ((lon / (2 * np.pi) + 0.5) * self.w) % self.w
        my = np.clip((0.5 - lat / np.pi) * self.h - 0.5, 0, self.h - 1)
        return mx.astype(np.float32), my.astype(np.float32)


class GoProMaxSource:
    """GoPro MAX .360: two HEVC streams, each holding 3 equi-angular cube faces plus two
    64 px lens-overlap strips. Stream 0 = left | front | right, stream 1 = down | back | up
    (stream 1 is rotated 90 degrees)."""

    def __init__(self, path, info):
        self.path, self.info = path, info
        self.raw_w, self.face = info["w"], info["h"]
        if self.raw_w != 3 * self.face + 64:
            raise ValueError(f"unexpected .360 stream size {self.raw_w}x{self.face}")
        f = self.face
        self.cut = f // 2 - 16  # width of the left/right tiles before each overlap strip
        self.w, self.h = 3 * f, 2 * f
        ramp = (np.arange(32, dtype=np.float32) + 0.5) / 32
        self.ramp = ramp[None, :, None]

    def _fix(self, raw):
        """Blend the 64 px overlap strips down to 32 px -> clean 3x2 EAC image."""
        c, W = self.cut, self.raw_w
        o1, o2 = c, W - c - 64

        def blend(x0):
            a = raw[:, x0:x0 + 32].astype(np.float32)
            b = raw[:, x0 + 32:x0 + 64].astype(np.float32)
            return (a * (1 - self.ramp) + b * self.ramp).astype(np.uint8)
        return np.concatenate([raw[:, :o1], blend(o1), raw[:, o1 + 64:o2], blend(o2), raw[:, o2 + 64:]], axis=1)

    def raw_frames(self, start, length):
        """Both streams stacked, overlap strips not yet blended."""
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(self.path),
               "-t", f"{length:.3f}", "-filter_complex", "[0:v:0][0:v:1]vstack[v]", "-map", "[v]",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        yield from pipe_frames(cmd, self.raw_w, 2 * self.face)

    def frames(self, start, length):
        for raw in self.raw_frames(start, length):
            yield self._fix(raw)

    def maps(self, x, y, z):
        f = self.face
        ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
        mx = np.zeros_like(x)
        my = np.zeros_like(x)
        eps = 1e-9
        # (condition, u, v, column, row): u/v are tangent-plane coords in [-1, 1],
        # u to the right and v downwards in the stored image.
        faces = [
            ((az >= ax) & (az >= ay) & (z > 0), x / (az + eps), -y / (az + eps), 1, 0),   # front
            ((ax >= ay) & (ax >= az) & (x > 0), -z / (ax + eps), -y / (ax + eps), 2, 0),  # right
            ((ax >= ay) & (ax >= az) & (x < 0), z / (ax + eps), -y / (ax + eps), 0, 0),   # left
            ((az >= ax) & (az >= ay) & (z < 0), y / (az + eps), -x / (az + eps), 1, 1),   # back
            ((ay >= ax) & (ay >= az) & (y < 0), -z / (ay + eps), -x / (ay + eps), 0, 1),  # down
            ((ay >= ax) & (ay >= az) & (y > 0), z / (ay + eps), -x / (ay + eps), 2, 1),   # up
        ]
        done = np.zeros(x.shape, bool)
        for cond, u, v, col, row in faces:
            m = cond & ~done
            done |= m
            eu = (4 / np.pi) * np.arctan(u[m])  # equi-angular warp
            ev = (4 / np.pi) * np.arctan(v[m])
            mx[m] = col * f + (eu + 1) / 2 * f - 0.5
            my[m] = row * f + (ev + 1) / 2 * f - 0.5
        return (np.clip(mx, 0, self.w - 1).astype(np.float32),
                np.clip(my, 0, self.h - 1).astype(np.float32))


def open_source(path):
    info = probe(path)
    if path.suffix.lower() == ".360":
        return GoProMaxSource(path, info)
    if abs(info["w"] / info["h"] - 2.0) > 0.05:
        raise ValueError(f"{info['w']}x{info['h']} is not equirectangular (2:1)")
    return EquirectSource(path, info)


def pipe_frames(cmd, w, h):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------- geometry

def equirect_rays(W, H):
    lon = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H, dtype=np.float32) + 0.5) / H * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)


def view_rays(out_w, out_h, hfov, yaw, pitch, cols=None, rows=None):
    """Ray directions for a flat camera pointed at (yaw, pitch) with horizontal fov hfov.
    cols/rows: optional subset of pixel coordinates to compute."""
    f = (out_w / 2) / math.tan(math.radians(hfov) / 2)
    cols = np.arange(out_w, dtype=np.float32) if cols is None else cols
    rows = np.arange(out_h, dtype=np.float32) if rows is None else rows
    u = (cols - out_w / 2 + 0.5) / f
    v = (rows - out_h / 2 + 0.5) / f
    x, y = np.meshgrid(u, -v)
    z = np.ones_like(x)
    n = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    p, t = math.radians(pitch), math.radians(yaw)
    y2 = y * math.cos(p) + z * math.sin(p)
    z2 = -y * math.sin(p) + z * math.cos(p)
    x3 = x * math.cos(t) + z2 * math.sin(t)
    z3 = -x * math.sin(t) + z2 * math.cos(t)
    return x3, y2, z3


def view_maps_equirect(out_w, out_h, hfov, yaw, pitch, eq_w, eq_h, step=8):
    """Fast perspective -> equirect lookup tables: exact on a coarse grid, then upsampled.
    Longitude is kept continuous around the view direction so upsampling never crosses the seam."""
    gx = np.arange(0, out_w + step, step, dtype=np.float32)
    gy = np.arange(0, out_h + step, step, dtype=np.float32)
    x, y, z = view_rays(out_w, out_h, hfov, yaw, pitch, cols=gx, rows=gy)
    lon = np.arctan2(x, z)
    t = math.radians(yaw)
    lon = (lon - t + np.pi) % (2 * np.pi) - np.pi + t
    lat = np.arcsin(np.clip(y, -1, 1))
    mx = ((lon / (2 * np.pi) + 0.5) * eq_w - 0.5).astype(np.float32)
    my = ((0.5 - lat / np.pi) * eq_h - 0.5).astype(np.float32)
    key = (out_w, out_h, step)
    if key not in _upsample_cache:
        c, r = np.meshgrid(np.arange(out_w, dtype=np.float32) / step, np.arange(out_h, dtype=np.float32) / step)
        _upsample_cache[key] = cv2.convertMaps(c, r, cv2.CV_16SC2)
    m1, m2 = _upsample_cache[key]
    mx = cv2.remap(mx, m1, m2, cv2.INTER_LINEAR)
    my = cv2.remap(my, m1, m2, cv2.INTER_LINEAR)
    cv2.min(cv2.max(my, 0), eq_h - 1, dst=my)
    return mx, my  # x may fall outside [0, eq_w): remap with BORDER_WRAP wraps it around


_upsample_cache = {}


def x_to_yaw(x, W):
    return x / W * 360.0 - 180.0


def y_to_pitch(y, H):
    return 90.0 - y / H * 180.0


def ang_dist(a, b):
    """Great-circle distance in degrees between (yaw, pitch) points."""
    y1, p1, y2, p2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(y1 - y2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ---------------------------------------------------------------- detection

def load_yolo(name):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("(ultralytics not installed - using motion detection only)")
        return None
    local = Path(__file__).resolve().parent / name  # reuse the copy next to this script
    return YOLO(str(local) if local.exists() else name)


def detect_people(model, frame_padded, W, imgsz):
    """Person boxes on a horizontally wrap-padded equirect frame -> list of (x1,y1,x2,y2,conf)."""
    if model is None:
        return []
    res = model.predict(frame_padded, classes=[0], conf=0.25, imgsz=imgsz, verbose=False)[0]
    boxes = [(float(a), float(b), float(c), float(d), float(e)) for (a, b, c, d), e in
             zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
    # drop duplicates of the same person seen on both sides of the seam
    boxes.sort(key=lambda b: -b[4])
    kept = []
    for b in boxes:
        cx = ((b[0] + b[2]) / 2) % W
        if all(abs(((((k[0] + k[2]) / 2) % W) - cx + W / 2) % W - W / 2) > 0.02 * W for k in kept):
            kept.append(b)
    return kept


def motion_blobs(mask_padded, min_area):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_padded)
    return [(float(x), float(y), float(x + w), float(y + h), float(a))
            for x, y, w, h, a in stats[1:] if a >= min_area]


def analyse(src, start, length, args, model):
    """Return per-frame (yaw, pitch, angular_height) of the skater, or None where not found."""
    W, H = ANALYSIS_W, ANALYSIS_H
    pad = W // 4
    fps = src.info["fps"]
    eq_mx, eq_my = src.maps(*equirect_rays(W, H))
    warm = min(2.0, start)  # extra seconds so the background model learns the scene
    warm_frames = int(round(warm * fps))
    stride = max(1, int(round(fps / 20)))  # look at ~20 frames per second; the gaps get interpolated
    bg = cv2.createBackgroundSubtractorMOG2(history=int(fps / stride * 3), varThreshold=32, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    track, moving_share, debug_frames = [], [], []
    prev, gap = None, 0
    total = int(length * fps)
    for i, raw in enumerate(src.frames(start - warm, length + warm)):
        if (i - warm_frames) % stride:
            if i >= warm_frames:
                track.append(None)
                moving_share.append(moving_share[-1] if moving_share else 0.0)
            continue
        frame = cv2.remap(raw, eq_mx, eq_my, cv2.INTER_LINEAR)
        mask = bg.apply(frame)
        if i < warm_frames:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        moving_share.append(cv2.countNonZero(mask) / mask.size)
        mask = cv2.dilate(mask, kernel, iterations=2)
        fpad = np.concatenate([frame, frame[:, :pad]], axis=1)
        mpad = np.concatenate([mask, mask[:, :pad]], axis=1)

        cands = []  # (yaw, pitch, angular height, score)
        for x1, y1, x2, y2, conf in detect_people(model, fpad, W, args.imgsz):
            roi = mpad[int(y1):int(y2), int(x1):int(x2)]
            moving = float((roi > 0).mean()) if roi.size else 0.0
            cx, cy = ((x1 + x2) / 2) % W, (y1 + y2) / 2
            cands.append((x_to_yaw(cx, W), y_to_pitch(cy, H), (y2 - y1) / H * 180.0,
                          conf * (0.25 + 2.0 * moving)))
        if not cands:  # no person found: fall back to the biggest moving thing
            for x1, y1, x2, y2, area in motion_blobs(mpad, min_area=W * H * 0.0004):
                # a skater stands on the ground: ignore things moving only up in the air (trees, flags)
                if y_to_pitch(y2, H) > 2 or y_to_pitch((y1 + y2) / 2, H) > 15:
                    continue
                cx, cy = ((x1 + x2) / 2) % W, (y1 + y2) / 2
                cands.append((x_to_yaw(cx, W), y_to_pitch(cy, H), (y2 - y1) / H * 180.0,
                              0.15 * min(1.0, area / (W * H * 0.005))))

        best = None
        for c in cands:
            score = c[3]
            if prev is not None:
                d = ang_dist(prev, c)
                if d > 12 + 6 * gap * stride:  # too far to be the same skater
                    continue
                score -= 0.01 * d
            if best is None or score > best[3]:
                best = c
        track.append(best[:3] if best else None)
        if best:
            prev, gap = best, 0
        else:
            gap += 1
        if args.debug:
            dbg = cv2.resize(frame, (W // 2, H // 2))
            small = cv2.resize(mask, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
            dbg[small, 2] = 255  # moving pixels tinted red
            if best:
                c = (int((best[0] + 180) / 360 * W / 2), int((90 - best[1]) / 180 * H / 2))
                cv2.circle(dbg, c, 12, (0, 255, 0), 3)
            for _ in range(stride):
                debug_frames.append(dbg)
        n = len(track)
        if n % 30 == 0:
            print(f"\r  finding skater {min(100, n / total * 100):5.1f}%", end="", flush=True)
    print("\r  finding skater 100.0%")
    return track, moving_share, debug_frames


# ---------------------------------------------------------------- smoothing

def fill_and_smooth(values, sigma_frames):
    """Linearly fill gaps (None), hold the ends, then gaussian-smooth (zero lag)."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    idx = np.arange(len(arr))
    ok = ~np.isnan(arr)
    if not ok.any():
        return None
    arr = np.interp(idx, idx[ok], arr[ok])
    if sigma_frames <= 0:
        return arr
    r = int(3 * sigma_frames)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(arr, r, mode="edge"), k, mode="valid")


def skater_speed(track, fps):
    """Skater speed in body heights per second (distance independent: a far-away skater moves
    fewer degrees but is also fewer degrees tall). Walking is ~1, skating is ~2-4."""
    if not any(track):
        return None
    unwrapped, last = [], None
    for t in track:
        if t is None:
            unwrapped.append(None)
            continue
        y = t[0] if last is None else last + ((t[0] - last + 180) % 360 - 180)
        unwrapped.append(y)
        last = y
    s = 0.12 * fps
    yaw = np.radians(fill_and_smooth(unwrapped, s))
    pitch = np.radians(fill_and_smooth([None if t is None else t[1] for t in track], s))
    height = np.radians(fill_and_smooth([None if t is None else t[2] for t in track], 2 * s))
    d = max(1, int(round(0.1 * fps)))  # measure over 0.1 s steps
    vec = np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=1)
    ang = np.zeros(len(track))
    ang[d:] = np.linalg.norm(vec[d:] - vec[:-d], axis=1) * fps / d  # sideways, rad/s
    rad = np.zeros(len(track))
    rad[d:] = np.abs(np.log(height[d:] / height[:-d])) * fps / d  # towards/away (1/s, relative)
    h = np.maximum(height, np.radians(2))
    speed = np.sqrt((ang / h) ** 2 + (rad / h) ** 2)
    # only trust the speed where the skater was tracked steadily (at least half of the checked
    # frames in the surrounding second); scattered one-off detections give fake speed spikes
    seen = np.array([t is not None for t in track], float)
    checked_per_sec = fps / max(1, int(round(fps / 20)))
    steady = np.convolve(seen, np.ones(int(fps)), mode="same") >= 0.5 * checked_per_sec
    speed[~steady] = 0
    return fill_and_smooth(list(speed), 0.25 * fps)


def camera_path(track, fps, args):
    # unwrap yaw so panning across the +-180 seam stays continuous
    unwrapped, last = [], None
    for t in track:
        if t is None:
            unwrapped.append(None)
            continue
        y = t[0] if last is None else last + ((t[0] - last + 180) % 360 - 180)
        unwrapped.append(y)
        last = y
    if last is None:
        return None
    yaw = fill_and_smooth(unwrapped, args.smooth * fps)
    pitch = fill_and_smooth([None if t is None else t[1] for t in track], args.smooth * 1.5 * fps)
    pitch = np.clip(pitch + args.tilt, -45, 30)
    aspect = args.width / args.height
    if args.fov == "auto":
        # vertical fov so the skater fills ~40% of frame height, converted to horizontal fov
        heights = fill_and_smooth([None if t is None else t[2] for t in track], 1.0 * fps)
        vfov = np.clip(heights / 0.40, 20, 120)
        hfov = np.degrees(2 * np.arctan(np.tan(np.radians(vfov) / 2) * aspect))
        hfov = np.clip(hfov, args.min_fov, args.max_fov)
    else:
        hfov = np.full(len(yaw), float(args.fov))
    return yaw, pitch, hfov


# ---------------------------------------------------------------- GPU render

class GpuView:
    """Cuts the flat view straight out of the source frame on the GPU (PyTorch), so nothing has to
    be unfolded into a big panorama on the CPU first."""

    def __init__(self, src, ow, oh):
        import torch
        self.t, self.src, self.ow, self.oh = torch, src, ow, oh
        self.dev = "cuda"
        u = (torch.arange(ow, device=self.dev, dtype=torch.float32) - ow / 2 + 0.5)
        v = (torch.arange(oh, device=self.dev, dtype=torch.float32) - oh / 2 + 0.5)
        self.v, self.u = torch.meshgrid(v, u, indexing="ij")
        self.gopro = isinstance(src, GoProMaxSource)
        if self.gopro:
            self.ramp = ((torch.arange(32, device=self.dev, dtype=torch.float32) + 0.5) / 32)[None, :, None]

    def _upload(self, frame):
        t = self.t
        x = t.from_numpy(frame).to(self.dev, non_blocking=True).float()
        if self.gopro:  # blend the lens-overlap strips (same as GoProMaxSource._fix)
            s, c, W = self.src, self.src.cut, self.src.raw_w
            o1, o2 = c, W - c - 64
            bl = lambda x0: x[:, x0:x0 + 32] * (1 - self.ramp) + x[:, x0 + 32:x0 + 64] * self.ramp
            x = t.cat([x[:, :o1], bl(o1), x[:, o1 + 64:o2], bl(o2), x[:, o2 + 64:]], 1)
        return x.permute(2, 0, 1)[None]  # 1x3xHxW

    def _rays(self, hfov, yaw, pitch):
        t = self.t
        f = (self.ow / 2) / math.tan(math.radians(hfov) / 2)
        x, y = self.u / f, -self.v / f
        z = t.ones_like(x)
        n = t.sqrt(x * x + y * y + 1)
        x, y, z = x / n, y / n, z / n
        p, q = math.radians(pitch), math.radians(yaw)
        y2 = y * math.cos(p) + z * math.sin(p)
        z2 = -y * math.sin(p) + z * math.cos(p)
        return x * math.cos(q) + z2 * math.sin(q), y2, -x * math.sin(q) + z2 * math.cos(q)

    def _coords(self, x, y, z, Wimg, Himg):
        t = self.t
        if not self.gopro:
            lon, lat = t.atan2(x, z), t.asin(t.clamp(y, -1, 1))
            return (lon / (2 * math.pi) + 0.5) * Wimg - 0.5, (0.5 - lat / math.pi) * Himg - 0.5
        f = self.src.face
        ax, ay, az = x.abs(), y.abs(), z.abs()
        eps = 1e-9
        mx, my = t.zeros_like(x), t.zeros_like(x)
        done = t.zeros_like(x, dtype=t.bool)
        faces = [
            ((az >= ax) & (az >= ay) & (z > 0), x / (az + eps), -y / (az + eps), 1, 0),
            ((ax >= ay) & (ax >= az) & (x > 0), -z / (ax + eps), -y / (ax + eps), 2, 0),
            ((ax >= ay) & (ax >= az) & (x < 0), z / (ax + eps), -y / (ax + eps), 0, 0),
            ((az >= ax) & (az >= ay) & (z < 0), y / (az + eps), -x / (az + eps), 1, 1),
            ((ay >= ax) & (ay >= az) & (y < 0), -z / (ay + eps), -x / (ay + eps), 0, 1),
            ((ay >= ax) & (ay >= az) & (y > 0), z / (ay + eps), -x / (ay + eps), 2, 1),
        ]
        for cond, uu, vv, col, row in faces:
            m = cond & ~done
            done |= m
            # stay 2 px inside the face: bicubic sampling must not reach into the neighbouring
            # stored face, which is not always the neighbour on the sphere (shows as thin lines)
            fx = (((4 / math.pi) * t.atan(uu) + 1) / 2 * f - 0.5).clamp(1.5, f - 2.5)
            fy = (((4 / math.pi) * t.atan(vv) + 1) / 2 * f - 0.5).clamp(1.5, f - 2.5)
            mx = t.where(m, col * f + fx, mx)
            my = t.where(m, row * f + fy, my)
        return mx, my

    def view(self, frame, hfov, yaw, pitch):
        t = self.t
        img = self._upload(frame)
        Himg, Wimg = img.shape[2], img.shape[3]
        mx, my = self._coords(*self._rays(hfov, yaw, pitch), Wimg, Himg)
        if not self.gopro:
            mx = t.remainder(mx, Wimg)
        grid = t.stack([(mx + 0.5) / Wimg * 2 - 1, (my + 0.5) / Himg * 2 - 1], -1)[None]
        out = t.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="border", align_corners=False)
        return out[0].permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------- render

def render(src, start, length, angles, out_path, args):
    yaw, pitch, hfov = angles
    ow, oh = args.width, args.height
    fps = src.info["fps"]
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps:.6f}", "-i", "-"]
    if has_audio(src.path):
        cmd += ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src.path),
                "-map", "0:v", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    cmd += ["-c:v", args.codec, "-pix_fmt", "yuv420p"]
    cmd += ["-cq", "19", "-preset", "p5"] if "nvenc" in args.codec else ["-crf", "18", "-preset", "veryfast"]
    cmd += ["-shortest", str(out_path)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = len(yaw)
    total = int(length * fps)
    if gpu_available():
        gv = GpuView(src, ow, oh)
        raw = src.raw_frames(start, length) if isinstance(src, GoProMaxSource) else src.frames(start, length)
        for i, frame in enumerate(raw):
            j = min(i, n - 1)
            enc.stdin.write(gv.view(frame, hfov[j], yaw[j], pitch[j]).tobytes())
            if i % 30 == 0:
                print(f"\r  rendering {min(100, i / total * 100):5.1f}%", end="", flush=True)
        enc.stdin.close()
        enc.wait()
        print("\r  rendering 100.0%")
        if enc.returncode != 0:
            raise RuntimeError("ffmpeg encoding failed")
        return
    # CPU: work on a full-quality equirect frame (GoPro cube faces are unfolded with a fixed table)
    if isinstance(src, EquirectSource):
        eq_w, eq_h, to_eq = src.w, src.h, None
    else:
        eq_w = int(round(src.face * 5.6 / 64)) * 64  # slightly denser than the source
        eq_h = eq_w // 2
        to_eq = src.maps(*equirect_rays(eq_w, eq_h))
    for i, frame in enumerate(src.frames(start, length)):
        j = min(i, n - 1)
        if to_eq is not None:
            frame = cv2.remap(frame, to_eq[0], to_eq[1], cv2.INTER_LINEAR)
        mx, my = view_maps_equirect(ow, oh, hfov[j], yaw[j], pitch[j], eq_w, eq_h)
        enc.stdin.write(cv2.remap(frame, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP).tobytes())
        if i % 30 == 0:
            print(f"\r  rendering {min(100, i / total * 100):5.1f}%", end="", flush=True)
    enc.stdin.close()
    enc.wait()
    print("\r  rendering 100.0%")
    if enc.returncode != 0:
        raise RuntimeError("ffmpeg encoding failed")


# ---------------------------------------------------------------- main

def process(path, args, model):
    t0 = time.time()
    src = open_source(path)
    info = src.info
    fps, dur = info["fps"], info["duration"] - args.end_trim
    print(f"  {fps:.2f} fps, {info['duration']:.1f}s long")
    if args.start is not None:
        start, length = args.start, min(args.seconds, dur - args.start)
        tail = 0.0
    else:
        # search the end of the clip for the trick (the walk back can take a while)
        tail = args.handling_search
        length = min(max(args.search, args.seconds + tail), dur)
        start = dur - length

    out_dir = Path(args.out) if args.out else path.parent / "reframed"
    out_dir.mkdir(parents=True, exist_ok=True)
    # reuse earlier detection results, so changing e.g. --seconds only re-renders
    cache = out_dir / ".cache" / f"{path.stem}_{start:.2f}_{length:.2f}_{path.stat().st_size}.npz"
    if cache.exists() and not args.debug and not args.no_cache:
        data = np.load(cache)
        track = [None if np.isnan(r[0]) else tuple(float(v) for v in r) for r in data["track"]]
        share, dbg = list(data["share"]), []
        print("  using saved detection results")
    else:
        track, share, dbg = analyse(src, start, length, args, model)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, track=np.array([t if t else (np.nan,) * 3 for t in track], float),
                 share=np.array(share, float))

    if tail > 0:
        # camera handling = a big part of the picture changing, and it keeps changing until the end
        # (a skater rolling right past the camera also spikes, but only briefly)
        n = len(track)
        cut = n
        thr = args.handling_threshold
        for i in range(max(0, n - int(tail * fps)), n):
            if share[i] > thr and float(np.mean(share[i:])) > thr * 0.8:
                cut = max(0, i - int(args.handling_pad * fps))
                print(f"  camera handling detected at {start + i / fps:.1f}s")
                break
        # the trick = the last stretch of real skating speed before that (walking back is slower)
        end = cut
        speed = skater_speed(track[:cut], fps)
        if speed is not None and args.skate_speed > 0:
            fast = speed > args.skate_speed
            min_run = int(0.4 * fps)
            i = cut - 1
            while i >= 0:
                if fast[i]:
                    j = i
                    while j >= 0 and fast[j]:
                        j -= 1
                    if i - j >= min_run:
                        end = min(cut, i + 1 + int(args.after * fps))
                        # don't linger on an empty scene if the skater was lost right after
                        seen = [k for k in range(j + 1, end) if track[k] is not None]
                        if seen:
                            end = min(end, max(i + 1, seen[-1] + int(0.7 * fps)))
                        print(f"  last skating ends at {start + (i + 1) / fps:.1f}s")
                        break
                    i = j
                i -= 1
            else:
                print("  no clear skating found; using the end of the clip")
        cut = end
        first = max(0, cut - int(args.seconds * fps))
        track, dbg = track[first:cut], dbg[first:cut]
        start, length = start + first / fps, (cut - first) / fps
    print(f"  using {start:.1f}s - {start + length:.1f}s")
    looked = len(track) // max(1, int(round(fps / 20)))
    print(f"  skater found in {sum(t is not None for t in track)}/{looked} checked frames "
          f"({time.time() - t0:.0f}s)")
    if dbg:
        dpath = out_dir / f"{path.stem}_tracking.mp4"
        vw = cv2.VideoWriter(str(dpath), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                             (dbg[0].shape[1], dbg[0].shape[0]))
        for f in dbg:
            vw.write(f)
        vw.release()
        print(f"  tracking preview: {dpath}")
    angles = camera_path(track, info["fps"], args)
    if angles is None:
        print("  could not find the skater; skipping")
        return
    out_path = out_dir / f"{path.stem}_reframed.mp4"
    render(src, start, length, angles, out_path, args)
    print(f"  saved {out_path}  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="Auto-reframe the end of 360 skate clips.")
    ap.add_argument("inputs", nargs="+", help=".360 / equirectangular .mp4 files, or folders")
    ap.add_argument("--seconds", type=float, default=8, help="length of each output clip in seconds (8)")
    ap.add_argument("--after", type=float, default=2.0, help="seconds to keep after the last skating (2.0)")
    ap.add_argument("--search", type=float, default=30, help="how far back from the end to look for the trick (30)")
    ap.add_argument("--skate-speed", type=float, default=0.8,
                    help="speed (body heights/s) that counts as skating; 0 = off, just use the end (0.8)")
    ap.add_argument("--end-trim", type=float, default=0, help="seconds to cut off the very end (0)")
    ap.add_argument("--handling-search", type=float, default=8,
                    help="look this many seconds at the end for you picking up the camera; 0 = off (8)")
    ap.add_argument("--handling-threshold", type=float, default=0.12,
                    help="share of the picture that must change to count as camera handling (0.12)")
    ap.add_argument("--handling-pad", type=float, default=1.0,
                    help="seconds to cut before the camera handling starts (1.0)")
    ap.add_argument("--start", type=float, help="use this start time (s) instead of counting from the end")
    ap.add_argument("--fov", default="auto", help="'auto' (zoom to skater) or horizontal degrees, e.g. 90")
    ap.add_argument("--min-fov", type=float, default=60, help="tightest auto zoom, degrees (60)")
    ap.add_argument("--max-fov", type=float, default=110, help="widest auto zoom, degrees (110)")
    ap.add_argument("--tilt", type=float, default=-3, help="extra tilt in degrees; negative looks lower (-3)")
    ap.add_argument("--smooth", type=float, default=0.35, help="pan smoothing in seconds (0.35)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--codec", default="auto", help="auto, h264_nvenc (NVIDIA GPU) or libx264 (CPU)")
    ap.add_argument("--model", default="yolo11s.pt", help="YOLO model (downloaded automatically)")
    ap.add_argument("--imgsz", type=int, default=1920, help="YOLO input size; larger finds smaller skaters")
    ap.add_argument("--no-ai", action="store_true", help="motion detection only, no YOLO")
    ap.add_argument("--out", help="output folder (default: 'reframed' next to each clip)")
    ap.add_argument("--debug", action="store_true", help="also save a tracking preview video")
    ap.add_argument("--no-cache", action="store_true", help="redo detection even if saved results exist")
    args = ap.parse_args()
    if args.fov != "auto":
        args.fov = float(args.fov)
    need("ffmpeg")
    need("ffprobe")
    args.codec = pick_codec(args.codec)

    files = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in (".360", ".mp4"))
        elif p.exists():
            files.append(p)
        else:
            print(f"not found: {p}")
    if not files:
        sys.exit("No input files.")
    model = None if args.no_ai else load_yolo(args.model)
    for f in files:
        print(f"\n{f.name}")
        try:
            process(f, args, model)
        except Exception as e:  # keep going with the other clips
            print(f"  failed: {e}")


if __name__ == "__main__":
    main()
