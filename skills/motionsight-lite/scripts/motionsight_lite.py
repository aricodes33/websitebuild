#!/usr/bin/env python3
"""CPU-only visual prompting for fine-grained video motion questions.

A GPU-free adaptation of MotionSight (https://github.com/NJU-PCALab/MotionSight):
the same two frame edits -- motion trails and object spotlighting -- are applied
to sampled frames and packed into labelled contact sheets that a vision model
(Claude) reads directly, replacing Qwen2.5-VL. Object boxes come from the model
looking at a gridded sheet instead of Grounding DINO + SAM2.

Subcommands:
  probe  VIDEO                         print fps, frame count, duration, size
  sheet  VIDEO [options]               write a contact sheet PNG
"""
import argparse
import json
import math
import os
import sys
from collections import deque

import cv2
import numpy as np


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video {path}")
    return cap


def video_info(path):
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": round(fps, 3), "frames": n, "duration_s": round(n / fps, 3), "width": w, "height": h}


def read_all_frames(path, start_idx, end_idx):
    """Read frames [start_idx, end_idx] sequentially (seeking is unreliable for many codecs)."""
    cap = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames = {}
    idx = start_idx
    while idx <= end_idx:
        ok, frame = cap.read()
        if not ok:
            break
        frames[idx] = frame
        idx += 1
    cap.release()
    return frames


def apply_trail(frames, idx, num_echos, decay_factor, echo_stride):
    """Blend earlier frames into frame idx, following MotionSight's apply_motion_trails."""
    history = deque(maxlen=num_echos)
    for k in range(num_echos - 1, -1, -1):
        j = idx - k * echo_stride
        if j in frames:
            history.append(frames[j])
    output = frames[idx].copy()
    current_decay = decay_factor
    for i in range(len(history) - 1, -1, -1):
        output = cv2.addWeighted(history[i], current_decay, output, 1 - current_decay, 0)
        current_decay *= decay_factor
    return output


def parse_boxes(specs):
    """Parse 'T:x1,y1,x2,y2' keyframes (T in seconds, coords normalised 0-1)."""
    keyframes = []
    for spec in specs:
        try:
            t, coords = spec.split(":")
            box = [float(v) for v in coords.split(",")]
            assert len(box) == 4
        except (ValueError, AssertionError):
            sys.exit(f"error: bad --box '{spec}', expected T:x1,y1,x2,y2")
        keyframes.append((float(t), box))
    return sorted(keyframes)


def box_at(keyframes, t):
    """Linearly interpolate the spotlight box at time t."""
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for (t0, b0), (t1, b1) in zip(keyframes, keyframes[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return [v0 + a * (v1 - v0) for v0, v1 in zip(b0, b1)]


def apply_spotlight(frame, boxes, blur_radius):
    """Blur everything outside the boxes, following MotionSight's apply_background_blur."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), 255, -1)
    k = blur_radius | 1
    blurred = cv2.GaussianBlur(frame, (k, k), k / 3)
    blurred = cv2.GaussianBlur(blurred, (k, k), k / 3)
    out = frame.copy()
    out[mask == 0] = blurred[mask == 0]
    return out


def draw_grid(frame):
    """Overlay a 0-1 coordinate grid (every 0.1) so boxes can be read off the image."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # Sized to stay legible once the frame is shrunk to a sheet tile.
    scale = w / 640
    thick = max(1, int(round(scale * 1.5)))
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(overlay, (x, 0), (x, h), (0, 255, 255), thick)
        cv2.line(overlay, (0, y), (w, y), (0, 255, 255), thick)
        cv2.putText(overlay, f".{i}", (x + 3, h - int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 255), thick + 1)
        cv2.putText(overlay, f".{i}", (3, y - int(4 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 255), thick + 1)
    return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)


def label(frame, text):
    scale = max(0.5, frame.shape[1] / 640)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (0, 0), (tw + 10, th + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (5, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return frame


def make_sheet(tiles, cols, tile_width):
    h, w = tiles[0].shape[:2]
    tile_h = int(h * tile_width / w)
    tiles = [cv2.resize(t, (tile_width, tile_h), interpolation=cv2.INTER_AREA) for t in tiles]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * tile_h + (rows - 1) * 4, cols * tile_width + (cols - 1) * 4, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (tile_h + 4), c * (tile_width + 4)
        sheet[y:y + tile_h, x:x + tile_width] = t
    return sheet


def cmd_probe(args):
    print(json.dumps(video_info(args.video), indent=2))


def cmd_sheet(args):
    info = video_info(args.video)
    fps, n = info["fps"], info["frames"]
    if n <= 0:
        sys.exit("error: video has no readable frames")
    start = max(0, int(args.start * fps))
    end = n - 1 if args.end is None else min(n - 1, int(args.end * fps))
    if end < start:
        sys.exit("error: --end is before --start")

    count = min(args.frames, end - start + 1)
    sample = np.linspace(start, end, count).round().astype(int).tolist()
    # Echoes default to a spacing that spans roughly the gap between sampled frames.
    stride = args.echo_stride or max(1, (sample[1] - sample[0]) // args.num_echos if count > 1 else 1)
    first_needed = max(0, sample[0] - (args.num_echos - 1) * stride) if args.trails else sample[0]
    frames = read_all_frames(args.video, first_needed, sample[-1])
    sample = [i for i in sample if i in frames]
    if not sample:
        sys.exit("error: could not decode the requested frames")

    keyframes = parse_boxes(args.box) if args.box else None
    tiles = []
    for idx in sample:
        t = idx / fps
        frame = apply_trail(frames, idx, args.num_echos, args.decay_factor, stride) if args.trails else frames[idx].copy()
        if keyframes:
            frame = apply_spotlight(frame, [box_at(keyframes, t)], args.blur_radius)
        if args.grid:
            frame = draw_grid(frame)
        tiles.append(label(frame, f"#{sample.index(idx) + 1}  t={t:.2f}s"))

    out = args.out or os.path.splitext(os.path.basename(args.video))[0] + "_sheet.png"
    cv2.imwrite(out, make_sheet(tiles, args.cols, args.tile_width))
    print(json.dumps({"sheet": os.path.abspath(out), "frames": [round(i / fps, 3) for i in sample],
                      "trails": args.trails, "echo_stride": stride if args.trails else None,
                      "spotlight": bool(keyframes)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="print video metadata")
    pr.add_argument("video")
    pr.set_defaults(func=cmd_probe)

    sh = sub.add_parser("sheet", help="write a labelled contact sheet")
    sh.add_argument("video")
    sh.add_argument("--frames", type=int, default=8, help="number of evenly spaced frames (default 8)")
    sh.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    sh.add_argument("--end", type=float, default=None, help="end time in seconds")
    sh.add_argument("--cols", type=int, default=4, help="tiles per row (default 4)")
    sh.add_argument("--tile-width", type=int, default=480, help="tile width in px (default 480)")
    sh.add_argument("--trails", action="store_true", help="add motion trails from preceding frames")
    sh.add_argument("--num-echos", type=int, default=7, help="trail length in frames (default 7)")
    sh.add_argument("--decay-factor", type=float, default=0.65, help="trail decay 0-1 (default 0.65)")
    sh.add_argument("--echo-stride", type=int, default=0, help="frames between echoes (default: auto)")
    sh.add_argument("--box", action="append", metavar="T:x1,y1,x2,y2",
                    help="spotlight keyframe, normalised coords; repeat to follow a moving object")
    sh.add_argument("--blur-radius", type=int, default=61, help="background blur kernel (default 61)")
    sh.add_argument("--grid", action="store_true", help="overlay a 0-1 coordinate grid")
    sh.add_argument("--out", help="output PNG path")
    sh.set_defaults(func=cmd_sheet)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
