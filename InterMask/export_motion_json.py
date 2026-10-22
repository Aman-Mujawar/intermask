"""
Export InterMask keypoint_npy outputs (frames, 22, 9) into a single compact
JSON file for the three.js 3D viewer (viewer/index.html).

Background (confirmed by reading utils/plot_script.py and utils/paramUtil.py
in github.com/gohar-malik/intermask, the upstream repo this project is based
on): preprocess_plot_motion() already writes one .npy per person per
generation, alongside the stick-figure .mp4:

    <name>_0.npy        person 1 (raw)
    <name>_1.npy        person 2 (raw)
    ik_<name>_0.npy      person 1 (foot-IK corrected)
    ik_<name>_1.npy      person 2 (foot-IK corrected)

each shaped (frames, 22, 9):
    [..., 0:3]  joint xyz position (meters, y = up)
    [..., 3:9]  6D continuous rotation encoding for that joint

This viewer only needs joint positions (that's all the paper's own
matplotlib renderer uses too -- it draws lines between joint positions along
the kinematic chain), so the rotation columns are read but dropped.

Usage -- run this once per generated clip you want to view in 3D, pointing
it at the keypoint_npy folder infer.py already wrote to:

    python export_motion_json.py \
        --npy_dir checkpoints/interhuman/trans_default/animation_infer/keypoint_npy \
        --name infer_prompts_ts10_cs4_topkr0.9_00_00 \
        --caption "In an intense boxing match, one is continuously punching while the other is defending and counterattacking." \
        --ik \
        --out viewer/motions/boxing.json

--name is the file's base name exactly as infer.py produced it, WITHOUT the
trailing _0 / _1 and without .npy -- i.e. drop only that suffix from an
actual file you see in keypoint_npy/.

This script only reads the .npy files infer.py / plot_script.py already
produce. It does not import, modify, or depend on anything else in the
repo (no torch, no CUDA, no conda env needed) -- it can run in any plain
Python 3 + numpy environment, so it can't affect your working generation
pipeline.
"""
import argparse
import json
import os
import numpy as np

# Mirrors utils/paramUtil.py:t2m_kinematic_chain exactly (22-joint HumanML3D
# skeleton used by interhuman). Kept as a literal copy rather than an
# import so this script has zero dependency on the rest of the repo.
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],       # right leg: pelvis -> right_hip -> right_knee -> right_ankle -> right_foot
    [0, 1, 4, 7, 10],       # left leg:  pelvis -> left_hip  -> left_knee  -> left_ankle  -> left_foot
    [0, 3, 6, 9, 12, 15],   # spine:     pelvis -> spine1 -> spine2 -> spine3 -> neck -> head
    [9, 14, 17, 19, 21],    # right arm: spine3 -> right_collar -> right_shoulder -> right_elbow -> right_wrist
    [9, 13, 16, 18, 20],    # left arm:  spine3 -> left_collar  -> left_shoulder  -> left_elbow  -> left_wrist
]

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def chain_to_edges(chain):
    edges = []
    for link in chain:
        for a, b in zip(link[:-1], link[1:]):
            edges.append([a, b])
    return edges


def load_person(npy_dir, name, person_idx, ik):
    prefix = "ik_" if ik else ""
    path = os.path.join(npy_dir, f"{prefix}{name}_{person_idx}.npy")
    arr = np.load(path)  # (frames, 22, 9)
    if arr.ndim != 3 or arr.shape[1:] != (22, 9):
        raise ValueError(
            f"{path} has shape {arr.shape}, expected (frames, 22, 9) -- "
            "this doesn't look like a keypoint_npy output from "
            "preprocess_plot_motion(). Double-check --npy_dir and --name."
        )
    positions = arr[..., :3]  # drop the 6D rotation columns, keep xyz only
    return positions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy_dir", required=True, help="Folder containing the *_0.npy / *_1.npy files (e.g. .../animation_infer/keypoint_npy)")
    ap.add_argument("--name", required=True, help="Base file name exactly as infer.py produced it, without the _0/_1 suffix or .npy extension")
    ap.add_argument("--caption", default="", help="Prompt text, shown as the on-screen title in the viewer")
    ap.add_argument("--ik", action="store_true", help="Use the foot-IK-corrected npy files (ik_ prefix) instead of the raw ones")
    ap.add_argument("--fps", type=int, default=30, help="Playback fps (30 matches infer.py's interhuman setting -- leave this unless you changed it)")
    ap.add_argument("--out", required=True, help="Output .json path, e.g. viewer/motions/boxing.json")
    args = ap.parse_args()

    person_a = load_person(args.npy_dir, args.name, 0, args.ik)
    person_b = load_person(args.npy_dir, args.name, 1, args.ik)

    frames = min(len(person_a), len(person_b))
    person_a, person_b = person_a[:frames], person_b[:frames]

    data = {
        "caption": args.caption,
        "fps": args.fps,
        "jointNames": JOINT_NAMES,
        "edges": chain_to_edges(T2M_KINEMATIC_CHAIN),
        "frames": frames,
        # 5 decimals keeps the JSON small (positions are in meters, so this
        # is ~0.01mm precision -- no visible quality loss).
        "personA": np.round(person_a, 5).tolist(),
        "personB": np.round(person_b, 5).tolist(),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out}  ({frames} frames, 2 people, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
