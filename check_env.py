"""
check_env.py -- run this inside the intermask_cpu conda env, from the repo
root, BEFORE `streamlit run streamlit_app/app.py`, any time you set up a new
machine/env or see the "Couldn't import the inference engine" banner.

It tries importing every third-party package the live inference path
(streamlit_app/inference_engine.py -> models/, utils/, data/, options/)
actually needs, and reports ALL missing ones in one pass -- instead of
hitting them one at a time through the Streamlit error banner, each
requiring its own round of "run it, read the banner, pip install, repeat."

Usage:
    conda activate intermask_cpu
    python check_env.py
"""
import importlib
import sys

# (import name, pip package name, why it's needed)
CHECKS = [
    ("torch", "torch", "core model framework"),
    ("numpy", "numpy", "array math"),
    ("scipy", "scipy", "filtering / spatial transforms"),
    ("einops", "einops==0.6.1", "tensor reshaping in the transformer"),
    ("clip", "git+https://github.com/openai/CLIP.git", "text encoder"),
    ("torchvision", "torchvision", "required transitively by the clip package"),
    ("positional_encodings", "positional-encodings==6.0.3", "transformer position encodings"),
    ("torch_geometric", "torch-geometric==2.3.1", "VQ-VAE's SAGEConv layers"),
    ("matplotlib", "matplotlib==3.3.1", "used by the save-to-disk / mp4 export path"),
    ("tqdm", "tqdm", "progress bars"),
    ("gdown", "gdown", "checkpoint downloader (prepare/download_models.py)"),
    ("streamlit", "streamlit==1.23.0", "the UI itself"),
]

print(f"Python: {sys.version}\n")
missing = []
for module_name, pip_name, why in CHECKS:
    try:
        importlib.import_module(module_name)
        print(f"  OK    {module_name:<24} ({why})")
    except ImportError as e:
        print(f"  MISSING {module_name:<22} ({why}) -- {e}")
        missing.append(pip_name)

print()
if missing:
    print(f"{len(missing)} package(s) missing. Install them with:\n")
    print(f"  pip install {' '.join(m for m in missing if not m.startswith('git+'))}")
    for m in missing:
        if m.startswith("git+"):
            print(f"  pip install {m}")
    sys.exit(1)
else:
    print("All required packages import cleanly. If the app still shows the "
          "'Couldn't import the inference engine' banner, the next most "
          "likely cause is a missing checkpoints/ folder, not a package.")
    sys.exit(0)
