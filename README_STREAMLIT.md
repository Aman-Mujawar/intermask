# InterMask Streamlit UI (Month 3 addition)

End-to-end: type or pick a prompt → real model runs on CPU → result renders
live in the same 3D viewer from Month 2, all inside one browser tab. No
manual `export_motion_json.py` step, no separate `python3 -m http.server`.

## What's in this folder

Everything here is **new**, additive to your existing pushed repo — nothing
in `infer.py`, `options/`, `models/`, or your existing `viewer/` is touched
or required to change.

```
streamlit_app/
  app.py                 # the Streamlit page — run this
  inference_engine.py    # loads the model ONCE, wraps infer.py's own logic
  viewer_component.py    # embeds the Month 2 3D viewer inside Streamlit
  sample_clips/
    sample.json          # synthetic placeholder, works with no model/checkpoints
  vendor/three/          # same vendored three.js from the Month 2 viewer
requirements_streamlit.txt
README_STREAMLIT.md      # this file
```

## How to merge into your repo

Drop the `streamlit_app/` folder and `requirements_streamlit.txt` straight
into the root of your existing `InterMask/` repo (the same level as
`infer.py`), then commit and push as usual:

```bash
# from inside your existing InterMask/ repo root
cp -r /path/to/streamlit_app .
cp /path/to/requirements_streamlit.txt .
git add streamlit_app requirements_streamlit.txt
git commit -m "Add Streamlit UI: live prompt -> 3D viewer demo"
git push
```

## Running it

Inside your existing `intermask_cpu` conda env (the same one `infer.py`
already runs in):

```bash
conda activate intermask_cpu
pip install -r requirements_streamlit.txt
streamlit run streamlit_app/app.py
```

Run it **from the repo root** (same folder as `infer.py`), not from inside
`streamlit_app/` — it imports `models/`, `utils/`, `options/`, `data/` the
same way `infer.py` does.

First page load triggers the model load (CLIP + both checkpoints) the first
time you click Generate — same cost `infer.py` pays every run, except here
it only happens once per server session, not once per prompt. After that,
each generation is just the ~10–15s CPU inference step.

If checkpoints aren't downloaded yet, or torch isn't on the path, the page
still loads fine and falls back to the synthetic `sample_clips/sample.json`
clip so you can confirm the viewer itself works before troubleshooting the
model side.

## What's genuinely new vs. Month 2

- **`inference_engine.py`** — this is the actual engineering addition.
  `infer.py` itself reloads CLIP and both checkpoints from scratch every
  time it runs, which is fine for a one-shot batch over `prompts.txt` but
  unusable for a UI where someone clicks Generate repeatedly. This module
  loads everything once (cached via `st.cache_resource` in `app.py`) and
  exposes two call paths:
  - `generate_preview()` — fast, in-memory, skips the matplotlib `.mp4`
    render and foot-IK pass, used for the live "Generate" button.
  - `generate_and_save()` — calls the repo's own unmodified
    `preprocess_plot_motion()`, i.e. byte-for-byte what `infer.py` itself
    produces (foot-IK corrected, saved to disk) — slower, meant for a
    "save this one for real" action rather than interactive browsing.
- **`viewer_component.py`** — same viewer as Month 2, but instead of reading
  a `.json` file over `fetch()` from a `python3 -m http.server`, it takes a
  motion dict already in memory and renders it directly via
  `st.components.v1.html`. three.js is still base64-inlined (not CDN), so
  it's exactly as offline-safe as before.
- **Side-by-side comparison tab** — the Month 3 feature from the plan,
  pulled forward: two independent prompt → generate → viewer pipelines in
  two columns. Not yet a single synchronized playhead (each clip's
  play/pause/scrub is independent) — flagged honestly in the app's own
  "About this demo" tab, not hidden.

## Honest limitations

- Still CPU-only — same ~10–15s/prompt as the Month 1/2 decks, no GPU
  hosting yet.
- This is a local `streamlit run`, not a public hosted demo link — that's
  still the open Milestone-2 item (Gradio/REST/HF Spaces), just now with a
  genuinely better front end sitting on top of it when you do get there.
- Side-by-side viewers are independent, not frame-synced.
