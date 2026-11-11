# Posture Coach

A private, fully local webcam posture coach. You calibrate *your own* relaxed, supported posture once. The app then compares live body and face landmarks with that baseline and gently alerts you when a deviation is **sustained**. It also nudges you to move and take breaks, because posture variety matters more than any single "perfect" position.

- **100% local.** Camera frames are analysed in memory on your CPU and never written to disk or sent anywhere. Only per-minute summary numbers are stored, in a local SQLite file. The dashboard is served on `127.0.0.1` only and loads nothing from the internet.
- **Built for head-on laptop and monitor cameras.** Uses MediaPipe Pose + Face Landmarker, with frontal-view measures that separate leaning toward the screen from head jutting forward.
- **Evidence-based.** Thresholds, messages and timing follow the ergonomics and posture literature. See [RESEARCH.md](RESEARCH.md) for sources and measured limitations.

## Quick start

```bash
uv venv -p 3.12 .venv
```

```bash
VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
```

```bash
.venv/bin/python -m posture_app fetch-models
```

```bash
.venv/bin/python -m posture_app run
```

`fetch-models` is a one-time download of Google's MediaPipe model files into `models/`, verified against pinned SHA-256 hashes. The app never downloads anything at runtime. Python 3.10–3.14 works; standard `pip` works too.

Then, in the dashboard that opens (default `http://127.0.0.1:8765`, or the next free port):

1. Sit in a relaxed, supported posture: back against the chair, shoulders relaxed (not pulled back), chin level, screen top at or slightly below eye level.
2. Click **Capture baseline** and hold still for about 5 s. Make a separate `standing` profile if you use a standing desk; the app switches profiles automatically.
3. Leave the tab open or close it. Tracking and desktop notifications continue while `run` is active.

## What it checks

| Measure | Default alert threshold (beyond your baseline) |
|---|---|
| Head dropped / slumping | 10% drop in eye-to-shoulder height |
| Head forward of shoulders | 10% (face magnified relative to shoulders) |
| Leaning toward screen | 15% closer |
| Looking down / chin up | 15° |
| Head tilted sideways | 10° |
| Leaning to one side | 7° (shoulder line) |

- **Thresholds:** each is widened automatically if your calibration showed more measurement noise. The *Gentle* preset scales thresholds ×1.3 and *Strict* ×0.8.
- **Alert timing:** an alert fires only after a deviation has lasted **30 s**, then at most every **3 min**. With *fade feedback*, the delay stretches up to 2× as your weekly score improves.
- **Movement and breaks:** you're reminded to move after **30 min** in one position, and to take a break after **50 min** at the screen.

All of these are adjustable in the dashboard.

## Commands

```
python -m posture_app run [--camera N] [--video FILE --loop] [--port P] [--model pose_lite|pose_full|pose_heavy]
python -m posture_app analyze FILE [--calibrate 0.5:5] [--profile sitting] [--csv out.csv]
python -m posture_app record FILE.mp4 [--speak]     # guided ~3-min labelled recording of good/bad postures
python -m posture_app evaluate FILE.mp4             # per-posture detection accuracy using the recording's labels
python -m posture_app fetch-models [--all]          # --all also fetches the large 'heavy' pose model
python -m posture_app doctor                        # checks models, camera and notifications
```

**Measuring accuracy on yourself:** `record` prompts you through good posture, slump, lean-in, forward head, looking down and side lean, and saves a `.labels.json` file next to the video. `evaluate` then reports, for each segment, how often the right issue was flagged (and, for good segments, how often nothing was flagged). Clips you record should stay private; they never leave your machine unless you move them.

## Testing

```bash
.venv/bin/python -m pytest
```

- `tests/test_core.py`: fast unit tests with synthetic landmarks and feature streams. They cover metric invariances, head-pose decomposition, calibration quality checks, hysteresis, sustained-alert timing, grace periods, cooldown, one-sided rules, lean-in vs. forward-head separation, yaw gating, movement and break reminders, and faded feedback.
- `tests/test_videos.py`: end-to-end runs on pre-recorded stock clips in `tests/videos/` (free-licensed Mixkit and Pexels footage). It also uses synthetic posture variants generated from those clips by `tests/tools/make_variants.py` (zoom → lean-in, head magnification → forward head, neck compression → slump, rotation → tilt). One case is a documented expected failure; see RESEARCH.md §4.

## Project layout

```
posture_app/
  landmarks.py    MediaPipe Pose + Face Landmarker, model verification, head-pose decomposition
  metrics.py      frontal-view posture features
  filters.py      One Euro smoothing
  rules.py        per-measure comparison vs. baseline, thresholds, coaching messages
  calibration.py  baseline capture, quality checks, profiles
  evaluator.py    hysteresis, sustained-alert timing, cooldown, faded feedback, movement/break reminders
  engine.py       processing loop, preview rendering, offline video analysis
  storage.py      local SQLite history (per-minute aggregates + events only)
  notifier.py     native notifications (notify-send / osascript / PowerShell)
  server.py       localhost dashboard server (Host check + per-launch token + CSP)
  web/            dashboard (no external resources)
models/           MediaPipe .task files (pinned SHA-256)
tests/            unit tests, video tests, test clips, variant generator
```

## Data and privacy

| What | Where | Contents |
|---|---|---|
| Settings | `~/.local/share/posture-coach/settings.json` (Linux), `~/Library/Application Support/posture-coach/` (macOS), `%APPDATA%\posture-coach\` (Windows) | Your preferences |
| Baselines | `baselines.json` in the same folder | Median feature values (numbers only) |
| History | `history.sqlite3` in the same folder | Per-minute good/bad seconds and events. **No images, no landmarks.** Clear it from the dashboard. |

Set `POSTURE_DATA_DIR` to use a different folder.

## Disclaimer

Posture Coach is a feedback and awareness tool, not a medical device. Evidence links some sustained postures (especially head and neck) with discomfort, but there's no single "correct" posture, and comfort plus regular movement matter most. If you have persistent pain, see a qualified health professional.
