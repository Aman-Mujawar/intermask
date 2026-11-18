"""
app.py -- InterMask Streamlit UI

End-to-end demo: pick (or type) a prompt, generate a two-person interaction
with the real pretrained InterMask model (CPU, via the patched conda env
from Month 1), and view it live in the 3D viewer built in Month 2 -- no
separate HTTP server, no exporting a .json by hand first.

Run from the repo root (same folder as infer.py), inside the intermask_cpu
conda env, after `pip install streamlit`:

    streamlit run streamlit_app/app.py

First load takes a while (loading CLIP + both checkpoints, same as infer.py).
After that, each generation is just the ~10-15s inference step, matching the
Month 1 CPU timing.
"""
from __future__ import annotations  # Python 3.7-safe: postpones annotation evaluation

import os
import sys
import time
import traceback

import streamlit as st


def _find_repo_root() -> str | None:
    """Locate the InterMask repo root (the folder containing infer.py and
    models/) by walking upward from this file's real location, instead of
    assuming a fixed folder depth. Some Streamlit versions/platforms don't
    resolve __file__ the way a plain `python script.py` run does, which is
    what broke the old fixed "two levels up" guess -- this version is
    self-correcting regardless of exact nesting or how the app was
    launched."""
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates.append(os.getcwd())

    seen = set()
    for start in candidates:
        d = start
        for _ in range(6):  # walk up at most 6 levels
            if d in seen:
                break
            seen.add(d)
            if os.path.isfile(os.path.join(d, "infer.py")) and os.path.isdir(os.path.join(d, "models")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is not None and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_STREAMLIT_APP_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.path.join(_REPO_ROOT or ".", "streamlit_app")
if _STREAMLIT_APP_DIR not in sys.path:
    sys.path.insert(0, _STREAMLIT_APP_DIR)

# viewer_component has no dependency on models/utils/etc., so import it
# directly (not as a `streamlit_app.` submodule -- that package-style import
# is what originally broke when the repo root wasn't found correctly).
import viewer_component as _vc
render_viewer, load_motion_json = _vc.render_viewer, _vc.load_motion_json

if _REPO_ROOT is None:
    ie = None
    _IE_IMPORT_ERROR = (
        "Could not locate the InterMask repo root (a folder containing both "
        "infer.py and models/) by searching upward from this file or from "
        "the current working directory. inference_engine.py needs that root "
        "on sys.path to import models/, utils/, data/, options/."
    )
else:
    try:
        import inference_engine as ie
        _IE_IMPORT_ERROR = None
    except Exception as _e:  # torch/CLIP/etc. not installed in this environment
        ie = None
        _IE_IMPORT_ERROR = _e

SAMPLE_CLIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_clips", "sample.json")

EXAMPLE_PROMPTS = [
    "In an intense boxing match, one is continuously punching while the other is defending and counterattacking.",
    "Two fencers engage in a thrilling duel, their sabres clashing and sparking as they strive for victory.",
    "Two people bow to each other.",
    "Two people are waving their hands and performing a dance step.",
    "The first person raises the right leg aggressively towards the second.",
]

st.set_page_config(page_title="InterMask 3D Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def get_engine():
    """Loads the model exactly once per server process (not once per click)
    -- this is the whole point of inference_engine.py existing as a module
    instead of calling infer.py as a subprocess."""
    return ie.InterMaskEngine(checkpoints_dir=os.path.join(_REPO_ROOT, "checkpoints"))


def engine_available() -> tuple[bool, str]:
    if ie is None:
        return False, (
            "Couldn't import the inference engine (likely means this isn't running "
            f"inside the `intermask_cpu` conda env): `{_IE_IMPORT_ERROR}`. "
            "Showing the synthetic sample clip below in the meantime -- the viewer "
            "itself works fine independent of this."
        )
    ckpt_dir = os.path.join(_REPO_ROOT, "checkpoints", "interhuman")
    if not os.path.isdir(ckpt_dir):
        return False, (
            f"No checkpoints found at `{ckpt_dir}`. Run "
            "`python prepare/download_models.py` first (same step as Month 1), "
            "then reload this page. Showing the synthetic sample clip below "
            "in the meantime."
        )
    return True, ""


def prompt_picker(label: str, key: str) -> str:
    choice = st.selectbox(label, EXAMPLE_PROMPTS + ["Custom..."], key=f"{key}_select")
    if choice == "Custom...":
        return st.text_input("Custom prompt", key=f"{key}_custom", placeholder="Two people ...")
    return choice


def generate_clip(prompt: str, engine) -> dict:
    gm = engine.generate_preview(prompt)
    # st.toast() needs Streamlit >=1.27 (Python >=3.8); pinned to 1.23.0 here
    # for Python 3.7 compatibility with the existing intermask_cpu env, so a
    # plain caption is used instead -- swap back to st.toast() if you ever
    # move off Python 3.7.
    st.caption(f"✓ Generated in {gm.gen_seconds:.1f}s (CPU)")
    # Stash the real, just-measured stats so the top-of-page stat chips (see
    # main()) reflect this server's actual loaded model and actual timing --
    # not the paper's numbers, which were GPU-measured and would be
    # misleading to show next to a CPU demo.
    st.session_state["model_param_str"] = engine.param_count_str
    st.session_state["last_gen_seconds"] = gm.gen_seconds
    return ie.motion_to_viewer_json(gm)


def stat_chips():
    """Params / last-generation-time / device chips, in the spirit of the
    project deck's 'Params 74M / GPU inference 0.77s' badges -- but reporting
    this server's own real numbers (CPU, whatever the loaded checkpoint
    actually contains) rather than the paper's GPU figures, so the demo never
    shows a number it didn't itself measure."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model size", st.session_state.get("model_param_str", "— (generate once)"))
    with c2:
        last = st.session_state.get("last_gen_seconds")
        st.metric("Last generation", f"{last:.2f}s" if last is not None else "—")
    with c3:
        st.metric("Device", "CPU")


def main():
    st.title("InterMask -- Live 3D Demo")
    st.caption(
        "Text prompt → CLIP → Inter-M Transformer → VQ-VAE decode → "
        "rendered live in the 3D viewer. Same CPU pipeline as the Monthly Progress decks, "
        "wrapped in a UI instead of a terminal command."
    )

    available, msg = engine_available()
    if not available:
        st.warning(msg)

    stat_chips()

    tab_single, tab_compare, tab_about = st.tabs(["Generate", "Side-by-side compare", "About this demo"])

    with tab_single:
        col_controls, col_view = st.columns([1, 2])
        with col_controls:
            st.subheader("Prompt")
            prompt = prompt_picker("Choose a prompt", "single")
            generate = st.button("Generate animation", type="primary", disabled=not available)
            st.divider()
            st.subheader("Or load a saved clip")
            st.caption("No model needed -- loads a clip already exported to streamlit_app/sample_clips/.")
            saved_files = _list_saved_clips()
            saved_choice = st.selectbox("Saved clip", ["(none)"] + saved_files, key="saved_single")
            load_saved = st.button("Load saved clip")

        with col_view:
            if generate and prompt:
                with st.spinner(f"Generating on CPU (~10-15s)... “{prompt[:60]}...”"):
                    try:
                        engine = get_engine()
                        motion = generate_clip(prompt, engine)
                        st.session_state["single_motion"] = motion
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.code(traceback.format_exc())
            elif load_saved and saved_choice != "(none)":
                motion = load_motion_json(os.path.join(os.path.dirname(SAMPLE_CLIP_PATH), saved_choice))
                st.session_state["single_motion"] = motion

            motion = st.session_state.get("single_motion")
            if motion is None:
                st.info("No clip generated yet -- showing the placeholder sample so you can confirm the viewer itself works.")
                motion = load_motion_json(SAMPLE_CLIP_PATH)
            render_viewer(motion, height=560)

    with tab_compare:
        st.caption(
            "Side-by-side prompt comparison -- generate two clips and view them in sync, "
            "the feature planned for Month 3 and pulled forward now that the single-clip "
            "pipeline is wired up end-to-end."
        )
        c1, c2 = st.columns(2)
        with c1:
            prompt_a = prompt_picker("Prompt A", "cmp_a")
        with c2:
            prompt_b = prompt_picker("Prompt B", "cmp_b")

        if st.button("Generate both", type="primary", disabled=not available):
            engine = get_engine()
            with st.spinner("Generating Prompt A..."):
                st.session_state["cmp_motion_a"] = generate_clip(prompt_a, engine) if prompt_a else None
            with st.spinner("Generating Prompt B..."):
                st.session_state["cmp_motion_b"] = generate_clip(prompt_b, engine) if prompt_b else None

        v1, v2 = st.columns(2)
        with v1:
            ma = st.session_state.get("cmp_motion_a")
            if ma:
                render_viewer(ma, height=460, accent_a="#ff8a3d", accent_b="#3da9ff")
            else:
                st.info("Generate to see Prompt A here.")
        with v2:
            mb = st.session_state.get("cmp_motion_b")
            if mb:
                render_viewer(mb, height=460, accent_a="#c45bff", accent_b="#5bffb0")
            else:
                st.info("Generate to see Prompt B here.")

    with tab_about:
        st.markdown(
            """
**What's running under the hood:**

- `inference_engine.py` wraps the repo's own `infer.py` model-loading and
  generation logic (same `RVQVAE`, `MaskTransformer`, `get_opt`,
  `MotionNormalizer` calls) so the model loads **once** per server process
  instead of once per click -- `infer.py` itself is unmodified.
- The fast "Generate" path skips the matplotlib `.mp4` render and foot-IK
  pass for interactive speed; `generate_and_save()` (not wired to a button
  here yet) calls the repo's real `preprocess_plot_motion()` for a
  foot-IK-corrected, disk-saved version identical to running `infer.py`
  directly.
- `viewer_component.py` inlines the same vendored three.js viewer built in
  Month 2 -- base64-embedded, zero CDN, zero separate HTTP server -- and
  renders it via `st.components.v1.html`.
- Checkpoints: `interhuman/vq_default` (519MB) and `interhuman/trans_default`
  (part of the ~1.7GB `prepare/download_models.py` pulls, which also fetches
  Inter-X checkpoints we don't use for inference -- see the Month 2 deck).

**Known limitations (same as the Month 2 deck's honest-status slide):**
- CPU-only: ~10-15s per generation, not yet GPU-hosted.
- No persistence between server restarts beyond what you explicitly save.
- Side-by-side comparison here is two independent clips, not a single
  synchronized playhead -- scrub/play on one doesn't drive the other yet.
            """
        )


def _list_saved_clips():
    d = os.path.dirname(SAMPLE_CLIP_PATH)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".json"))


if __name__ == "__main__":
    main()
