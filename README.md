# Auto-edit for GoPro MAX skate sessions

Finds **every trick** in your GoPro MAX `.360` clips and cuts each one into a short 16:9 clip (about 4 s) that follows you. Everything runs on your own PC and graphics card, with no internet or AI service, so it costs nothing to use.

There are two ways to use it:

- **Reviewed (recommended):** the tools find candidates and make picture sheets; each attempt is judged by eye (landed, fall, or simple fail / just rolling), and only landed tricks and falls are rendered, each with its own start and end. Session 46 part 2 went from 133 candidates to 25 landed tricks plus 2 falls, with everything kept being relevant.
- **Fully automatic:** `autoedit.bat` cuts a fixed ~4 s clip around every likely trick. It's faster but keeps failed attempts and plain rolling.

## Reviewed workflow

Run these with `%USERPROFILE%\autoreframe-env\Scripts\python.exe`, where `<clips>` is a folder of `.360` files:

1. **Scan** (once per clip, about as long as the footage):
   `scan.py <clips>`
2. **Candidates and quick sheets:**
   `review_strips.py <clips> <review> --threshold 0.3`
   This writes `candidates.json` (number, clip, time, score) and `sheet_XX.jpg` (7 frames per candidate). Use it to drop the obvious non-tricks (walking, board flying away).
3. **Detail sheets** for the rest, from 3 s before to 7 s after each candidate in 0.5 s steps, with tight framing:
   `review_detail.py <clips> <review> <detail> 3,9,14,... -3:7:0.5`
   For each candidate, decide:
   - **Landed:** you ride away on the board for at least about 1 s.
   - **Fall:** you end up on the ground.
   - **Drop it:** you step off or the board flies away, or there's no trick (just rolling or pushing).

   Note where the trick really starts (before the pop or the approach onto the ledge) and end the clip about 2 s after the **last** landing. Check the ride-away for a second, back-to-back trick: `review_strips.py` uses `--gap 1.5` by default so back-to-back tricks get their own rows, but still check the ride-away.
4. **Write `jobs.json`** with start and end in source seconds, e.g. `[{"n": 24, "start": 92.6, "end": 99.8, "label": "landed"}]`, then render:
   `render_selected.py <clips> <review> <clips>\tricks_v1 <review>\jobs.json`
5. **Trim the ride-away tails** the way you trim by hand:
   `trim_ends.py <clips> <review> <clips>\tricks_v1`
   Each clip ends once the rider has shrunk to below 25% of their size at the trick, but never sooner than 1 s after it. The rule (`end_rule.json`) was fitted to the LosslessCut trims of session 46 part 2, where it lands within 0.6 s of the hand cuts on average (0.9 s before). Falls and long grinds (over 7 s) keep their judged end. Add `--dry-run` to only print the new lengths.
6. **Join** the clips into one video. Re-encode once while joining: after `trim_ends.py`'s stream-copy trims, a plain `-c copy` join leaves small timestamp glitches at a few seams.

   ```bash
   ls *.mp4 | sort | sed "s/.*/file '&'/" > list.txt
   ffmpeg -f concat -safe 0 -i list.txt -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k ../tricks.mp4
   ```

`examples/part2/` has the real `candidates.json` and `jobs.json` from session 46 part 2.

## Fully automatic

Drag a session folder (or some `.360` files) onto **autoedit.bat**, or run:

```bat
autoedit.bat "D:\Videos\skate\session"
```

The clips appear in a `tricks` folder next to the `.360` files, named `<clip>_<time>s_<score>.mp4`. A higher score means the trick is more likely one you'd keep. Delete the ones you don't want.

**Keep the laptop plugged in.** On battery, Windows slows the GPU and CPU down 2–3×.

## How it works

1. **Scan (once per clip):** the whole clip is checked at 10 frames per second for people and skateboards (YOLO on your GPU), plus motion and sound. The results are saved in `.autoreframe\` next to the clips, so later runs skip this step. It takes about as long as the clip itself.
2. **Find you:** only people standing on a skateboard count, so walkers and scooter riders are ignored.
3. **Find tricks:** every moment where you hit peak speed on the board is scored by a small model trained on **your own edits** (`model.json`). It looks at approach speed, whether you were on the board before and after, the landing sound, and similar signals.
4. **Cut:** each likely trick gets a clip from 1.9 s before to 2.3 s after the peak, matching how you cut your own edits.

## Accuracy (tested against your 69 hand-made edits)

Each clip was tested with a model that hadn't seen it:

| `--threshold` | Your tricks found | Extra clips (25 files) |
|---|---|---|
| 0.3 | 59 / 69 | 87 |
| **0.4 (default)** | **58 / 69** | **74** |
| 0.5 | 51 / 69 | 62 |
| 0.7 | 34 / 69 | 26 |

Most extra clips are other attempts at the same trick, which look almost identical to the one you kept.

## Options

| Option | What it does |
|---|---|
| `--threshold 0.5` | Fewer, surer clips (misses more). `0.3` gives more clips |
| `--before 2.5` / `--after 3` | More room before or after the trick |
| `--list` | Only print the trick times, don't render |
| `--fov 90` | Fixed zoom instead of auto zoom |
| `--tilt -8` | Aim lower (negative) or higher (positive) |
| `--smooth 0.6` | Smoother, slower camera moves |
| `--out D:\edits` | Put the clips somewhere else |

## Teaching it more

When you hand-edit more sessions in GoPro Player, keep the exports (they keep the GoPro timecode) and retrain:

```bat
%USERPROFILE%\autoreframe-env\Scripts\python.exe train_model.py "<folder with .360 files>" "<folder with your edits>" [more pairs...]
```

Edits are matched to their `.360` by file name (`GS016712 (2).mp4` → `GS016712.360`) and by timecode. The script prints how well the new model does, then saves `model.json`. More examples, especially of the kind of trick you keep, make it better.

## Files

- `autoedit.py`: the main script (scan, find tricks, render)
- `scan.py`: the one-time whole-clip scan
- `tricks.py`: trick finding and the model
- `train_model.py`: learns from your edits
- `review_strips.py`, `review_detail.py`: picture sheets for judging each attempt
- `render_selected.py`: renders the chosen candidates (`landed=1,4 fall=3` or a `jobs.json` with per-clip start and end)
- `autoreframe.py`: 360 → flat video rendering (GPU). Run on its own, it's the older "last N seconds" mode.
- `yolo11s.pt` isn't in the repo; ultralytics downloads it on first use.

## Setup (already done on this PC)

Python 3.12 and ffmpeg (winget), plus a Python environment in `%USERPROFILE%\autoreframe-env` with PyTorch (CUDA 12.8), ultralytics, opencv-python and numpy. To set it up on another PC:

```bat
winget install Python.Python.3.12 Gyan.FFmpeg
python -m venv %USERPROFILE%\autoreframe-env
%USERPROFILE%\autoreframe-env\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
%USERPROFILE%\autoreframe-env\Scripts\pip install ultralytics opencv-python numpy
```

Updating the NVIDIA driver (to version 610 or newer) lets ffmpeg encode on the GPU too, which makes rendering faster.
