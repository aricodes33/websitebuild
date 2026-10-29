---
name: motionsight-lite
description: Answer fine-grained motion questions about a video file (direction of movement, camera pan/zoom/tilt, gestures, which object moved first, speed changes, small actions) by preprocessing frames with motion trails and object spotlighting, then reading them as images. GPU-free adaptation of MotionSight. Use when the user gives a video (mp4, mov, avi, webm) and asks how something in it moves, or when plain frame sampling misses subtle motion.
---

# MotionSight Lite

A CPU-only version of [MotionSight](https://github.com/NJU-PCALab/MotionSight). MotionSight boosts a
video model's motion understanding with two visual prompts: **motion trails** (earlier frames blended
in as fading echoes, so movement leaves a visible streak) and **object spotlighting** (the background
blurred away around the object being asked about). This skill applies the same edits with OpenCV and
packs the results into labelled contact sheets that you read yourself — no Qwen, Grounding DINO or SAM2.

Requires Python 3 with `opencv-python` and `numpy` (`pip install opencv-python numpy`).

The script is `scripts/motionsight_lite.py` next to this file. Below, `MS` means
`python <this skill's directory>/scripts/motionsight_lite.py`.

## Workflow

1. **Probe the video.** `MS probe VIDEO` prints fps, frame count, duration and size.

2. **Get an overview.** `MS sheet VIDEO --frames 8 --grid --out overview.png`, then view the PNG.
   Each tile is labelled with its number and timestamp. Use it to find *when* the relevant motion
   happens and *where* the object is. The yellow grid marks normalised coordinates in steps of 0.1.

3. **Pick the visual prompt that fits the question.**
   - Camera motion (pan, tilt, zoom, shake, static): `--trails` on the whole frame, no spotlight.
     A static camera shows a crisp background with streaks only on moving things; a pan smears the
     whole scene sideways; a zoom smears outward/inward from the centre.
   - Object motion (direction, gesture, which way it turns, what it does): `--trails` plus `--box`
     spotlighting on that object.
   - Ordering or counting questions ("which moved first", "how many times"): no trails, more frames
     over a narrower `--start`/`--end` window.

4. **Zoom in on the moment.** Narrow to the interval found in step 2, e.g.
   `MS sheet VIDEO --start 6.9 --end 8.2 --frames 6 --cols 3 --trails --out detail.png`.

5. **Spotlight the object** when the question is about one object. Read its box off the gridded
   overview at two or more timestamps and pass each as `--box T:x1,y1,x2,y2` (seconds, then
   normalised left, top, right, bottom). Boxes are linearly interpolated between keyframes and held
   past the ends, so:
   - give keyframes that cover the whole `--start`/`--end` window, including the last frame;
   - make boxes generous (pad ~20%) — the trail extends behind the object, and a tight box clips it;
   - add more keyframes if the object speeds up, turns, or changes size quickly.
   Re-render with `--grid` to check the box follows the object before relying on it. Repeat `--box`
   sets are for one object; for two objects, render two sheets.

6. **Answer from the evidence.** Cite the tiles you used ("tiles #3–#5 show the car's trail
   pointing up-screen, so it moves toward the camera"). If the sheets are ambiguous, say so and
   render a narrower window rather than guessing.

## Reading trails

Trails are drawn from frames *before* each tile's timestamp. The crisp copy is where the object is
now; fainter echoes are where it was. So motion runs **from the faint end toward the crisp end**.
Long trails mean fast motion. `--num-echos` (default 7) sets trail length and `--echo-stride`
(default: auto, spanning the gap between tiles) sets the spacing; lower the stride for fast motion
that turns into a blur, raise it for slow motion that barely shows.

## Options for `sheet`

| Option | Default | Purpose |
|---|---|---|
| `--frames N` | 8 | evenly spaced frames in the window |
| `--start S` / `--end S` | whole video | time window in seconds |
| `--cols N` | 4 | tiles per row |
| `--tile-width PX` | 480 | tile width; raise for small objects, lower to fit more tiles |
| `--trails` | off | add motion trails |
| `--num-echos N` | 7 | trail length |
| `--decay-factor F` | 0.65 | how fast echoes fade (0–1) |
| `--echo-stride N` | auto | frames between echoes |
| `--box T:x1,y1,x2,y2` | none | spotlight keyframe; repeatable |
| `--blur-radius N` | 61 | background blur strength |
| `--grid` | off | coordinate grid overlay |
| `--out PATH` | `<video>_sheet.png` | output image |

The command prints JSON with the sheet path and the timestamps of every tile.

## Limits

- Keep sheets to about 12 tiles; more tiles means smaller tiles and lost detail. Render several
  sheets for long videos instead.
- Boxes are rectangles you estimate by eye, so spotlighting is rougher than MotionSight's SAM2
  masks; overlapping objects inside one box are not separated.
- This is an adaptation: MotionSight's published benchmark scores were measured with Qwen2.5-VL and
  do not transfer to this skill.
