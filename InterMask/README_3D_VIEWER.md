# InterMask 3D Motion Viewer

Replaces the flat, fixed-camera stick-figure `.mp4` with a real 3D view the
professor can orbit/zoom around during your demo. Built from the actual
joint-position data `infer.py` already writes — no model changes, no
retraining, and nothing in the existing pipeline was touched.

Confirmed directly from the upstream repo (`utils/plot_script.py` and
`utils/paramUtil.py`): each `keypoint_npy` file is `(frames, 22, 9)` —
3 values/joint = real XYZ position, 6 = a rotation encoding the viewer
doesn't need (positions alone fully define the pose, same as the paper's
own matplotlib renderer). 22-joint skeleton, 30 fps, two people per clip.

## Files

```
export_motion_json.py   -- run this after generating a clip
viewer/
  index.html             -- the 3D viewer itself, open this in a browser
  motions/sample.json     -- synthetic placeholder clip (already works --
                             open the viewer now, before plugging in any
                             real data, to confirm it runs on your machine)
  vendor/three/           -- three.js, vendored locally (no internet
                             needed to run this -- important for demoing
                             live without depending on a CDN being up)
```

## 1. Generate a clip as usual

Nothing changes here — run `infer.py` exactly like you already do. This
writes `keypoint_npy/<name>_0.npy` and `<name>_1.npy` (plus `ik_`-prefixed
foot-corrected versions) like it always has.

## 2. Export it for the 3D viewer

```bash
python export_motion_json.py \
  --npy_dir checkpoints/interhuman/trans_default/animation_infer/keypoint_npy \
  --name infer_prompts_ts10_cs4_topkr0.9_00_00 \
  --caption "In an intense boxing match, one is continuously punching while the other is defending and counterattacking." \
  --ik \
  --out viewer/motions/boxing.json
```

- `--name` is the file's base name exactly as it appears in `keypoint_npy/`,
  with the trailing `_0`/`_1` and `.npy` dropped.
- `--ik` uses the smoother foot-IK-corrected version (recommended — it's
  what slide 6 of the Month 1 deck used too).
- This only reads `.npy` files that already exist; it needs plain Python 3
  + numpy, nothing else (no torch, no conda env, no CUDA).

Repeat once per clip you want viewable (e.g. once for boxing, once for
fencing, for the side-by-side comparison feature).

## 3. Open the viewer

Browsers block loading local JSON over `file://`, so serve the folder:

```bash
cd viewer
python3 -m http.server 8000
```

Then open:

```
http://localhost:8000/index.html?file=motions/sample.json      (placeholder -- try this first)
http://localhost:8000/index.html?file=motions/boxing.json      (your real clip, after step 2)
```

Controls: drag to orbit, scroll to zoom, right-drag to pan, play/pause and
a frame scrubber at the bottom, speed selector (0.25x–2x — handy for
pointing out a specific moment to the professor in slow motion).

## Notes

- `sample.json` is a synthetic placeholder (procedurally animated, not a
  real model output) — it exists only so you can confirm the viewer itself
  works on your machine before you've generated anything. Swap it for a
  real exported clip for the actual presentation.
- For the "side-by-side prompt comparison" feature: easiest route is two
  browser tabs/windows side by side, each loaded with a different
  `?file=...clip.json`, paused at a comparable moment — no extra code
  needed for that part. Say the word if you'd rather have one page showing
  both clips at once and I'll build that next.
