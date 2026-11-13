"""
tricks.py - find your tricks in a scanned clip (see scan.py).

  scan data -> person tracks -> candidate moments (you, on a board, at peak speed)
  -> features -> trick score (logistic model trained on your own edits, model.json)
"""
import json
import math
from pathlib import Path

import numpy as np

PERSON, BOARD = 0, 36
FEATS = ["speed", "sp_before", "sp_after", "board_before", "board_after", "lift", "audio", "audio_after",
         "log_h", "track_share", "moving", "color_dist", "log_t_end", "seen_after", "seen_before",
         "ride_after", "board_sep", "follow_seen", "follow_speed"]
MODEL = Path(__file__).resolve().parent / "model.json"


# ---------------------------------------------------------------- scan data

def load_scan(path):
    d = np.load(path)
    det = d["dets"]
    W, H = float(d["W"]), float(d["H"])
    s = {"fps": float(d["fps"]), "duration": float(d["duration"]), "share": d["share"], "audio": d["audio"],
         "n": int(len(d["share"]))}
    s["frame"] = det[:, 0].astype(int)
    s["cls"] = det[:, 1].astype(int)
    s["yaw"] = (det[:, 2] + det[:, 4]) / 2 / W * 360 - 180
    s["pitch"] = 90 - (det[:, 3] + det[:, 5]) / 2 / H * 180
    s["bottom"] = 90 - det[:, 5] / H * 180
    s["h"] = (det[:, 5] - det[:, 3]) / H * 180
    s["w"] = (det[:, 4] - det[:, 2]) / W * 360
    s["conf"] = det[:, 6]
    s["moving"] = det[:, 7]
    s["color"] = det[:, 8:14]
    s["onb"] = on_board(s)
    s["byf"] = {}
    for i, f in enumerate(s["frame"]):
        s["byf"].setdefault(f, []).append(i)
    return s


def yaw_diff(a, b):
    return (a - b + 180) % 360 - 180


def angd(s, i, j):
    return math.hypot(yaw_diff(s["yaw"][i], s["yaw"][j]) * math.cos(math.radians(s["pitch"][j])),
                      s["pitch"][i] - s["pitch"][j])


def on_board(s):
    """For each person detection: is a skateboard right under their feet?"""
    res = np.zeros(len(s["cls"]), bool)
    by_frame = {}
    for b in np.where(s["cls"] == BOARD)[0]:
        by_frame.setdefault(s["frame"][b], []).append(b)
    for i in np.where(s["cls"] == PERSON)[0]:
        for b in by_frame.get(s["frame"][i], []):
            if (abs(yaw_diff(s["yaw"][b], s["yaw"][i])) < max(s["w"][i] * 0.8, 1.5)
                    and s["bottom"][i] - 0.35 * s["h"][i] < s["pitch"][b] < s["bottom"][i] + 0.25 * s["h"][i]):
                res[i] = True
    return res


def build_tracks(s, max_gap=10):
    """Greedy frame-to-frame linking of person detections. Returns track id per detection (-1 = board)."""
    tid = np.full(len(s["cls"]), -1)
    active, next_id, used = {}, 0, set()
    persons = np.where(s["cls"] == PERSON)[0]
    for i in persons[np.lexsort((-s["conf"][persons], s["frame"][persons]))]:
        f = s["frame"][i]
        best, best_d = None, 1e9
        for t, j in active.items():
            gap = f - s["frame"][j]
            if gap <= 0 or gap > max_gap or (f, t) in used:
                continue
            d = angd(s, i, j) / max(s["h"][j], s["h"][i], 3.0)
            d += abs(math.log(max(s["h"][i], 0.5) / max(s["h"][j], 0.5))) * 0.5
            if d < (0.6 + 0.15 * gap) and d < best_d:
                best, best_d = t, d
        if best is None:
            best, next_id = next_id, next_id + 1
        tid[i] = best
        active[best] = i
        used.add((f, best))
        for t in [t for t, j in active.items() if f - s["frame"][j] > max_gap]:
            del active[t]
    return tid


def follow(s, start, frames, step=1):
    """Follow a person from detection 'start' forwards (step=1) or backwards (step=-1) through any
    person detections (tracks break when you pass right by the camera). -> [(frame, det or None)]"""
    cur, out = start, []
    f0 = s["frame"][start]
    for k in range(1, frames + 1):
        f = f0 + k * step
        best, bd = None, 1e9
        for j in s["byf"].get(f, []):
            if s["cls"][j] != PERSON:
                continue
            d = angd(s, j, cur) / max(s["h"][cur], s["h"][j], 3)
            if d < bd:
                best, bd = j, d
        gap = abs(f - s["frame"][cur])
        if best is not None and bd < 0.8 + 0.3 * gap:
            cur = best
            out.append((f, best))
        else:
            out.append((f, None))
    return out


# ---------------------------------------------------------------- candidates and features

def smooth(x, sig):
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def fill(x):
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return None
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


def after_features(s, det, f, fps):
    """What happens 1-3.5 s after the moment: still riding (landed) or board gone (bail)?"""
    path = follow(s, det, int(3.5 * fps))
    late = [(fr, j) for fr, j in path if fr >= f + fps]
    seen = [j for _, j in late if j is not None]
    ride = float(np.mean([s["onb"][j] for j in seen])) if seen else 0.0
    sep = 0
    for fr, j in late:
        if j is None or s["onb"][j]:
            continue
        if any(s["cls"][b] == BOARD and angd(s, b, j) < 4 * max(s["h"][j], 3) for b in s["byf"].get(fr, [])):
            sep += 1
    pts = [(fr, j) for fr, j in path if j is not None]
    mv = 0.0
    if len(pts) > 3:
        d = sum(angd(s, pts[k + 1][1], pts[k][1]) / max(s["h"][pts[k][1]], 2) for k in range(len(pts) - 1))
        mv = d / ((pts[-1][0] - pts[0][0]) / fps + 1e-6)
    return [ride, sep / max(1, len(late)), len(seen) / max(1, len(late)), mv]


def find_candidates(s):
    """Moments where a person on a skateboard reaches a local speed peak. -> list of dicts."""
    tid = build_tracks(s)
    n, fps = s["n"], s["fps"]
    a = np.pad(s["audio"][:n].astype(float), (0, max(0, n - len(s["audio"]))))
    med = np.array([np.median(a[max(0, i - 50):i + 50]) for i in range(n)]) + 1e-6
    az = a / med
    persons = s["cls"] == PERSON
    board_total = np.bincount(tid[persons & s["onb"]], minlength=tid.max() + 1).astype(float)
    me_col = np.nanmedian(s["color"][persons & s["onb"]], 0) if (persons & s["onb"]).any() else np.full(6, np.nan)
    out = []
    for t in np.unique(tid[tid >= 0]):
        idx = np.where(tid == t)[0]
        if len(idx) < 8 or s["onb"][idx].sum() < 2:
            continue
        f0, f1 = s["frame"][idx].min(), s["frame"][idx].max() + 1
        if f1 - f0 < 12:
            continue
        L = f1 - f0
        ser = {k: np.full(L, np.nan) for k in ("yaw", "pitch", "h", "bottom", "board", "moving")}
        for i in idx:
            r = s["frame"][i] - f0
            for k in ("yaw", "pitch", "h", "bottom", "moving"):
                ser[k][r] = s[k][i]
            ser["board"][r] = s["onb"][i]
        yaw = fill(ser["yaw"])
        if yaw is None:
            continue
        yaw = np.degrees(np.unwrap(np.radians(yaw)))
        pitch, h, bottom = fill(ser["pitch"]), fill(ser["h"]), fill(ser["bottom"])
        seen = ~np.isnan(ser["h"])
        board = np.nan_to_num(ser["board"])
        ys, ps, hs = np.radians(smooth(yaw, 1.5)), np.radians(smooth(pitch, 1.5)), np.radians(smooth(h, 3))
        vec = np.stack([np.cos(ps) * np.sin(ys), np.sin(ps), np.cos(ps) * np.cos(ys)], 1)
        sp = np.zeros(L)
        sp[1:] = np.linalg.norm(vec[1:] - vec[:-1], axis=1) * fps / np.maximum(hs[1:], np.radians(2))
        sp = smooth(sp, 3)
        sp[np.convolve(seen.astype(float), np.ones(10) / 10, mode="same")[:L] < 0.5] = 0

        def win(x, i, a_, b_):
            a2, b2 = max(0, i + a_), min(L, i + b_)
            return x[a2:b2] if b2 > a2 else np.array([0.0])

        for i in range(L):
            lo, hi = max(0, i - 15), min(L, i + 16)
            if sp[i] < 0.5 or sp[i] < sp[lo:hi].max():
                continue
            if not seen[lo:hi].any() or board[lo:hi][seen[lo:hi]].mean() < 0.2:
                continue
            f = f0 + i
            base = np.median(win(bottom, i, -20, 21))
            lift = np.max((win(bottom, i, -10, 11) - base) / np.maximum(win(h, i, -10, 11), 1))
            sb, sa = seen[max(0, i - 15):i].sum(), seen[i + 5:i + 21].sum()
            bb = board[max(0, i - 15):i][seen[max(0, i - 15):i]].mean() if sb else 0
            ba = board[i + 5:i + 21][seen[i + 5:i + 21]].mean() if sa else 0
            near = idx[(s["frame"][idx] >= f - 20) & (s["frame"][idx] <= f + 20)]
            col = np.nanmedian(s["color"][near], 0)
            cd = float(np.nan_to_num(np.linalg.norm((col - me_col)[[1, 2, 4, 5]]), nan=30))
            det = idx[np.argmin(np.abs(s["frame"][idx] - f))]
            feats = [sp[i], win(sp, i, -15, 0).mean(), win(sp, i, 1, 16).mean(), bb, ba, lift,
                     az[max(0, f - 10):f + 11].max(), az[f:f + 16].max(), math.log(max(h[i], 1)),
                     board_total[t] / max(1, board_total.sum()), np.nanmean(ser["moving"]), cd,
                     math.log1p(max(0, s["duration"] - f / fps)), sa / 16, sb / 15]
            feats += after_features(s, det, f, fps)
            out.append({"t": f / fps, "det": int(det), "track": int(t), "feats": feats})
    return out


# ---------------------------------------------------------------- model

def fit(X, y, l2=1.0, it=3000, lr=0.1):
    X = np.nan_to_num(np.asarray(X, float))
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    sw = np.where(y > 0, (1 - y.mean()) / y.mean(), 1.0)  # balance classes
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": b, "feats": FEATS}


def score(model, feats):
    X = np.nan_to_num(np.asarray(feats, float))
    z = (X - np.array(model["mu"])) / np.array(model["sd"])
    return 1 / (1 + np.exp(-(z @ np.array(model["w"]) + model["b"])))


def load_model():
    return json.load(open(MODEL))


def handling_mask(s, cands, still=0.05, need=2.0):
    """Candidates during camera setup (before the picture first stays still for 2 s) or pickup
    (after it last does). Skating right past the camera mid-clip is not affected."""
    share, fps = np.asarray(s["share"]), s["fps"]
    k = int(need * fps)
    calm = np.convolve((share < still).astype(float), np.ones(k), mode="valid") >= k  # calm[i]: i..i+k still
    if not calm.any():
        return np.zeros(len(cands), bool)
    first = np.argmax(calm) / fps
    last = (len(calm) - 1 - np.argmax(calm[::-1]) + k) / fps
    return np.array([c["t"] < first or c["t"] > last for c in cands], bool)


def select(cands, scores, thr, gap=3.0):
    """Best candidate per 3 s neighbourhood above the threshold, in time order."""
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        if scores[i] < thr:
            break
        if all(abs(cands[i]["t"] - cands[j]["t"]) > gap for j in keep):
            keep.append(i)
    return sorted(keep, key=lambda i: cands[i]["t"])


# ---------------------------------------------------------------- camera path for rendering

def follow_path(s, det, t0, t1):
    """(yaw, pitch, h) per scan frame for [t0, t1] following the person from 'det' both ways."""
    fps, f = s["fps"], s["frame"][det]
    fa, fb = int(math.floor(t0 * fps)), int(math.ceil(t1 * fps))
    pts = {f: det}
    for fr, j in follow(s, det, max(0, fb - f)):
        if j is not None:
            pts[fr] = j
    for fr, j in follow(s, det, max(0, f - fa), step=-1):
        if j is not None:
            pts[fr] = j
    frames = np.arange(fa, fb + 1)
    out = []
    for key in ("yaw", "pitch", "h"):
        x = np.full(len(frames), np.nan)
        for k, fr in enumerate(frames):
            if fr in pts:
                x[k] = s[key][pts[fr]]
        if key == "yaw":
            ok = ~np.isnan(x)
            x[ok] = np.degrees(np.unwrap(np.radians(x[ok])))
        x = fill(x) if (~np.isnan(x)).sum() >= 2 else np.full(len(frames), s[key][det])
        out.append(x)
    return frames / fps, out
