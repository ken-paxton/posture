"""Command-line entry point.

  python -m posture_app run [--camera N | --video FILE] [--port 8765]
  python -m posture_app analyze FILE [--calibrate 0.5:5] [--csv out.csv]
  python -m posture_app record FILE [--camera N]     guided, labelled test recording
  python -m posture_app evaluate FILE                accuracy against a recording's labels
  python -m posture_app fetch-models                 one-time model download + verification
  python -m posture_app doctor                       check camera, models, notifications
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
import urllib.request
import webbrowser
from collections import Counter
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import Settings, data_dir  # noqa: E402

log = logging.getLogger("posture")


# ---------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    from .calibration import BaselineStore
    from .engine import Engine
    from .server import App, serve
    from .sources import CameraSource, VideoSource
    from .storage import History

    settings = Settings.load()
    if args.fps:
        settings.analysis_fps = args.fps
    if args.model:
        settings.pose_model = args.model
    source = (VideoSource(args.video, realtime=True, loop=args.loop) if args.video
              else CameraSource(args.camera if args.camera is not None else settings.camera))
    history = None if args.video else History()
    store = BaselineStore()
    engine = Engine(settings, source, store=store, history=history, notify=not args.no_notify)
    if history:
        history.profile = settings.active_profile
    app = App(engine, history, settings)
    httpd, _ = serve(app, args.port)
    engine.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"Posture Coach running at {url}  (Ctrl+C to quit)")
    print(f"Data folder: {data_dir()}")
    if not store.profiles:
        print("No baseline yet: open the dashboard and click 'Capture baseline'.")
    if not args.no_browser:
        webbrowser.open(url)

    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        while not stop["flag"]:
            time.sleep(0.3)
            if args.video and not args.loop and engine.snapshot().get("state") == "stopped":
                print("Video finished; dashboard stays up until Ctrl+C.")
                args.loop = True  # print once
    finally:
        httpd.shutdown()
        engine.stop()
        if history:
            history.close()
        settings.save()
    return 0


# ---------------------------------------------------------------------------- analyze
def _parse_window(s: str) -> tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def cmd_analyze(args) -> int:
    from .calibration import BaselineStore
    from .engine import analyze_video

    settings = Settings.load()
    if args.delay is not None:
        settings.alert_delay_s = args.delay
    settings.fade_feedback = False
    baseline = None
    if args.profile:
        baseline = BaselineStore().profiles.get(args.profile)
        if baseline is None:
            print(f"No saved profile '{args.profile}'.", file=sys.stderr)
            return 2
    r = analyze_video(args.file, settings=settings, calibrate=_parse_window(args.calibrate), baseline=baseline,
                      fps=args.fps)
    tl = r["timeline"]
    scored = [x for x in tl if x["score"] is not None]
    print(f"{args.file}: {len(tl)} frames analysed, {r['perf_ms']:.1f} ms/frame")
    print("calibration:", json.dumps(r["calibration"], indent=None))
    if scored:
        bad = sum(x["bad"] for x in scored) / len(scored)
        print(f"out-of-range share: {bad:.0%}; mean score {sum(x['score'] for x in scored) / len(scored):.0f}")
        c = Counter(i for x in scored for i in x["issues"])
        for k, n in c.most_common():
            print(f"  {k:15s} {n / len(scored):.0%} of frames")
    for e in r["events"]:
        print(f"  t={e['t']:7.1f}s  {e['kind']:14s} {', '.join(e['issues'])}")
    if args.csv:
        import csv

        keys = sorted({k for x in tl for k in x["features"]})
        rule_keys = sorted({k for x in tl for k in x["rules"]})
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "score", "bad", "issues"] + keys + [f"dev_{k}" for k in rule_keys])
            for x in tl:
                w.writerow([x["t"], x["score"], int(x["bad"]), "|".join(x["issues"])]
                           + [x["features"].get(k) for k in keys] + [x["rules"].get(k) for k in rule_keys])
        print("wrote", args.csv)
    return 0


# ---------------------------------------------------------------------------- record / evaluate
RECORD_SCRIPT = [
    # (seconds, label, instruction). Label None = transition, excluded from scoring.
    (8, "good", "Sit in your relaxed, supported, comfortable posture (this is used as the baseline)."),
    (12, "good", "Stay in good posture; work, type or look around naturally."),
    (3, None, "Now slump: let your back round and your head sink."),
    (12, "slump", "Hold the slump."),
    (3, None, "Return to good posture."),
    (10, "good", "Good posture."),
    (3, None, "Now lean your whole body toward the screen."),
    (12, "lean_in", "Hold the lean."),
    (3, None, "Return to good posture."),
    (8, "good", "Good posture."),
    (3, None, "Now poke your head forward (chin forward), keeping your back where it is."),
    (12, "forward_head", "Hold the forward head."),
    (3, None, "Return to good posture."),
    (8, "good", "Good posture."),
    (3, None, "Now look down at your lap / phone."),
    (12, "head_flexion", "Hold looking down."),
    (3, None, "Return to good posture."),
    (8, "good", "Good posture."),
    (3, None, "Now lean to one side (shoulders tilted)."),
    (12, "shoulder_tilt", "Hold the side lean."),
    (3, None, "Return to good posture."),
    (8, "good", "Good posture. Done!"),
]


def cmd_record(args) -> int:
    import cv2

    from .sources import CameraSource

    out = Path(args.file)
    src = CameraSource(args.camera, args.width, args.height)
    got = src.read(timeout=5)
    if got is None:
        print("Camera produced no frames.", file=sys.stderr)
        src.close()
        return 1
    _, frame, _ = got
    h, w = frame.shape[:2]
    fps = args.fps
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    segments, t = [], 0.0
    for dur, label, text in RECORD_SCRIPT:
        segments.append({"start": t, "end": t + dur, "label": label, "instruction": text})
        t += dur
    total = t
    print(f"Recording {total:.0f}s to {out} at {w}x{h}. Follow the prompts.\n")
    start = time.monotonic()
    seg_i, seq, next_frame_t = -1, -1, 0.0
    try:
        while True:
            el = time.monotonic() - start
            if el >= total:
                break
            i = next(k for k, s in enumerate(segments) if s["start"] <= el < s["end"])
            if i != seg_i:
                seg_i = i
                s = segments[i]
                tag = s["label"] or "move"
                print(f"[{el:5.1f}s] ({tag:>13s}) {s['instruction']}")
                if shutil.which("spd-say") and args.speak:
                    os.system(f"spd-say -w {json.dumps(s['instruction'])} >/dev/null 2>&1 &")
            got = src.read(seq, timeout=1.0)
            if got is None:
                continue
            seq, frame, _ = got
            # Write frames at a constant rate so timestamps in the file match the script.
            while next_frame_t <= el:
                vw.write(frame)
                next_frame_t += 1.0 / fps
    finally:
        vw.release()
        src.close()
    labels = out.with_suffix(".labels.json")
    labels.write_text(json.dumps({"video": out.name, "calibrate": [1.0, 7.0], "segments": segments}, indent=2))
    print(f"\nSaved {out} and {labels}. Evaluate with: python -m posture_app evaluate {out}")
    return 0


def evaluate_against_labels(video: str, labels: dict, settings: Settings | None = None) -> dict:
    from .engine import analyze_video

    settings = settings or Settings()
    settings.alert_delay_s = 1e9  # judge frame-level detection, not alert timing
    settings.fade_feedback = False
    settings.movement_reminder_min = settings.break_reminder_min = 0
    r = analyze_video(video, settings=settings, calibrate=tuple(labels.get("calibrate", (1.0, 7.0))))
    rows = []
    for seg in labels["segments"]:
        if not seg["label"]:
            continue
        # Skip the first 2 s of each segment (people take time to settle into it).
        frames = [x for x in r["timeline"] if seg["start"] + 2 <= x["t"] < seg["end"] and x["score"] is not None]
        if not frames:
            rows.append({**seg, "frames": 0, "hit_rate": None})
            continue
        if seg["label"] == "good":
            hit = sum(not x["bad"] for x in frames) / len(frames)
        else:
            hit = sum(seg["label"] in x["issues"] for x in frames) / len(frames)
        other = Counter(i for x in frames for i in x["issues"])
        rows.append({**seg, "frames": len(frames), "hit_rate": hit, "flagged": dict(other.most_common(3))})
    return {"calibration": r["calibration"], "segments": rows}


def cmd_evaluate(args) -> int:
    video = Path(args.file)
    lab_path = Path(args.labels) if args.labels else video.with_suffix(".labels.json")
    labels = json.loads(lab_path.read_text())
    res = evaluate_against_labels(str(video), labels, Settings.load())
    print("calibration:", res["calibration"]["phase"] if res["calibration"] else None)
    print(f"{'segment':>14s} {'time':>13s} {'frames':>6s} {'correct':>8s}  most flagged")
    by_label: dict[str, list[float]] = {}
    for s in res["segments"]:
        hr = s["hit_rate"]
        if hr is not None:
            by_label.setdefault(s["label"], []).append(hr)
        print(f"{s['label']:>14s} {s['start']:5.0f}-{s['end']:<5.0f}s {s['frames']:6d} "
              f"{'' if hr is None else f'{hr:7.0%}':>8s}  {s.get('flagged', '')}")
    print("\nsummary (good = share of frames NOT flagged; others = share flagged with that issue):")
    for k, v in by_label.items():
        print(f"  {k:14s} {sum(v) / len(v):.0%}")
    return 0


# ---------------------------------------------------------------------------- models / doctor
def cmd_fetch_models(args) -> int:
    from .landmarks import MODEL_SHA256, MODEL_URLS, MODELS_DIR, sha256

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    names = [n for n in MODEL_URLS if args.all or "heavy" not in n]
    for name in names:
        path = MODELS_DIR / name
        if path.exists() and sha256(path) == MODEL_SHA256[name]:
            print(f"ok       {name}")
            continue
        print(f"download {name} …")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
        digest = sha256(tmp)
        if digest != MODEL_SHA256[name]:
            tmp.unlink()
            print(f"  SHA-256 mismatch for {name} ({digest}); not installed.", file=sys.stderr)
            return 1
        tmp.replace(path)
        print(f"ok       {name}")
    print("Models are stored locally; the app never downloads anything at runtime.")
    return 0


def cmd_doctor(args) -> int:
    from .landmarks import MODEL_FILES, model_path

    ok = True
    for key, name in MODEL_FILES.items():
        try:
            model_path(name)
            print(f"[ok]   model {name}")
        except Exception as e:  # noqa: BLE001
            print(f"[{'warn' if 'heavy' in key else 'FAIL'}] model {name}: {e}")
            ok = ok and "heavy" in key
    try:
        from .sources import CameraSource

        cam = CameraSource(args.camera)
        got = cam.read(timeout=5)
        cam.close()
        if got:
            print(f"[ok]   camera {args.camera}: {got[1].shape[1]}x{got[1].shape[0]}")
        else:
            print(f"[FAIL] camera {args.camera} opened but produced no frames")
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] camera {args.camera}: {e}")
        ok = False
    if sys.platform.startswith("linux"):
        print(f"[{'ok' if shutil.which('notify-send') else 'warn'}]   notify-send "
              f"{'found' if shutil.which('notify-send') else 'missing (install libnotify-bin for desktop alerts)'}")
    print(f"[info] data folder: {data_dir()}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="posture-coach", description="Local, private webcam posture coach.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start tracking and the local dashboard (default)")
    r.add_argument("--camera", type=int)
    r.add_argument("--video", help="use a pre-recorded video instead of the camera")
    r.add_argument("--loop", action="store_true", help="loop the video")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--fps", type=float)
    r.add_argument("--model", choices=["pose_lite", "pose_full", "pose_heavy"])
    r.add_argument("--no-browser", action="store_true")
    r.add_argument("--no-notify", action="store_true")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="run the pipeline over a video and summarise")
    a.add_argument("file")
    a.add_argument("--calibrate", default="0.5:5", help="baseline window START:END seconds (default 0.5:5)")
    a.add_argument("--profile", help="use a saved baseline profile instead")
    a.add_argument("--delay", type=float, help="alert delay seconds (default from settings)")
    a.add_argument("--fps", type=float)
    a.add_argument("--csv", help="write per-frame features and deviations")
    a.set_defaults(func=cmd_analyze)

    rec = sub.add_parser("record", help="record a guided, labelled test clip from the camera")
    rec.add_argument("file")
    rec.add_argument("--camera", type=int, default=0)
    rec.add_argument("--width", type=int, default=1280)
    rec.add_argument("--height", type=int, default=720)
    rec.add_argument("--fps", type=float, default=15.0)
    rec.add_argument("--speak", action="store_true", help="read prompts aloud (spd-say)")
    rec.set_defaults(func=cmd_record)

    ev = sub.add_parser("evaluate", help="measure detection accuracy on a labelled recording")
    ev.add_argument("file")
    ev.add_argument("--labels")
    ev.set_defaults(func=cmd_evaluate)

    fm = sub.add_parser("fetch-models", help="download and verify the model files (one time)")
    fm.add_argument("--all", action="store_true", help="include the large 'heavy' pose model")
    fm.set_defaults(func=cmd_fetch_models)

    d = sub.add_parser("doctor", help="check models, camera and notifications")
    d.add_argument("--camera", type=int, default=0)
    d.set_defaults(func=cmd_doctor)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if not args.cmd:
        args = p.parse_args(["run"] + (argv or sys.argv[1:]))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
