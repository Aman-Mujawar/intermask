"""
inference_engine.py

A refactored, importable version of infer.py's model-loading + generation
logic. infer.py itself is NOT modified -- this module calls the exact same
classes (RVQVAE, MaskTransformer) and helpers (get_opt, fixseed,
MotionNormalizer) the same way infer.py's __main__ block and gen_motions()
already do, verified against this repo's own source.

Why this exists instead of just shelling out to `python infer.py` per prompt:
infer.py loads CLIP + both checkpoints fresh every time it runs (meant for a
one-shot batch job over prompts.txt). In a Streamlit app that would mean
re-paying that load cost (which is most of the wall-clock time) on every
single click. InterMaskEngine loads everything ONCE and is cached across
reruns via @st.cache_resource in app.py, so only the actual generation step
(~10-15s on CPU, matching the Month 1 numbers) happens per prompt.

Two generation paths are exposed:
  - generate_preview(): fast, in-memory only, skips the matplotlib .mp4
    render and the foot-IK pass. Good enough for interactive use in the
    3D viewer, and is the normal path the Streamlit UI uses.
  - generate_and_save(): calls the repo's own, unmodified
    utils.plot_script.preprocess_plot_motion() -- the exact function
    infer.py itself calls -- so the saved .npy / .mp4 outputs are
    byte-for-byte consistent with running infer.py directly, including the
    foot-IK correction. Slower (includes the matplotlib render); used by
    the "Save this clip to disk" button, not for live preview.
"""
import os
import time
from dataclasses import dataclass
from os.path import join as pjoin

import numpy as np
import scipy.ndimage.filters as filters
import torch

from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer
from utils.get_opt import get_opt
from utils.utils import fixseed
from data.utils import MotionNormalizer

CLIP_VERSION = "ViT-L/14@336px"
DIM_POSE = 12


@dataclass
class GeneratedMotion:
    joints_a: np.ndarray   # (frames, 22, 3) xyz, Person A
    joints_b: np.ndarray   # (frames, 22, 3) xyz, Person B
    caption: str
    fps: int
    gen_seconds: float


def _motion_to_viewer_dict(gm: GeneratedMotion) -> dict:
    """Same schema export_motion_json.py writes: caption, fps, jointNames,
    edges, frames, personA, personB -- so render_viewer() from
    viewer_component.py accepts either a live GeneratedMotion or a .json
    file saved earlier, interchangeably."""
    from utils import paramUtil

    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    ]
    edges = []
    for chain in paramUtil.t2m_kinematic_chain:
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append([a, b])

    return {
        "caption": gm.caption,
        "fps": gm.fps,
        "frames": gm.joints_a.shape[0],
        "jointNames": joint_names,
        "edges": edges,
        "personA": np.round(gm.joints_a, 5).tolist(),
        "personB": np.round(gm.joints_b, 5).tolist(),
    }


class InterMaskEngine:
    """Loads the InterHuman VQ-VAE + Mask Transformer checkpoints once and
    keeps them resident for repeated generation calls. Instantiate this
    through app.py's @st.cache_resource-wrapped factory, not directly per
    request."""

    def __init__(self, checkpoints_dir="checkpoints", dataset_name="interhuman",
                 trans_name="trans_default", which_epoch="best_fid", gpu_id=-1):
        self.device = torch.device("cpu" if gpu_id == -1 else f"cuda:{gpu_id}")
        self.dataset_name = dataset_name
        self.fps = 30

        trans_opt_path = pjoin(checkpoints_dir, dataset_name, trans_name, "opt.txt")
        main_opt = get_opt(trans_opt_path, self.device)
        fixseed(main_opt.seed)

        vq_opt_path = pjoin(checkpoints_dir, dataset_name, main_opt.vq_name, "opt.txt")
        vq_opt = get_opt(vq_opt_path, self.device)
        main_opt.num_tokens = vq_opt.nb_code
        main_opt.code_dim = vq_opt.code_dim

        self.main_opt = main_opt
        self.vq_opt = vq_opt
        self.normalizer = MotionNormalizer()

        # ---- VQ-VAE, same construction as infer.py's load_vq_model() ----
        self.net = RVQVAE(
            vq_opt, DIM_POSE, vq_opt.nb_code, vq_opt.code_dim, vq_opt.code_dim,
            vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth,
            vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm,
        )
        vq_ckpt_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "model", "best_fid.tar")
        vq_ckpt = torch.load(vq_ckpt_path, map_location="cpu")
        model_key = "vq_model" if "vq_model" in vq_ckpt else "net"
        missing, unexpected = self.net.load_state_dict(vq_ckpt[model_key], strict=False)
        assert len(unexpected) == 0, f"Unexpected VQ-VAE keys: {unexpected}"
        self.net = self.net.to(self.device).eval()
        self.vq_epoch = vq_ckpt.get("ep", "?")

        # ---- Mask Transformer, same construction as infer.py's load_trans_model() ----
        self.trans = MaskTransformer(
            code_dim=main_opt.code_dim, cond_mode="text", latent_dim=main_opt.latent_dim,
            ff_size=main_opt.ff_size, num_layers=main_opt.n_layers, num_heads=main_opt.n_heads,
            dropout=main_opt.dropout, clip_dim=768, cond_drop_prob=main_opt.cond_drop_prob,
            clip_version=CLIP_VERSION, opt=main_opt,
        )
        model_dir = pjoin(checkpoints_dir, dataset_name, trans_name, "model")
        ckpt_file = next((f for f in os.listdir(model_dir) if which_epoch in f), None)
        if ckpt_file is None:
            raise FileNotFoundError(
                f"No checkpoint matching '{which_epoch}' in {model_dir} -- "
                f"run prepare/download_models.py first."
            )
        trans_ckpt = torch.load(pjoin(model_dir, ckpt_file), map_location=self.device)
        model_key = "t2m_transformer" if "t2m_transformer" in trans_ckpt else "trans"
        missing, unexpected = self.trans.load_state_dict(trans_ckpt[model_key], strict=False)
        assert len(unexpected) == 0, f"Unexpected transformer keys: {unexpected}"
        self.trans = self.trans.to(self.device).eval()
        self.trans_epoch = trans_ckpt.get("ep", "?")

        # Total trainable-parameter count across both networks, computed once
        # at load time from the real loaded weights (not hardcoded from the
        # paper) -- shown in the UI next to generation time, mirroring the
        # "Params 74M / GPU inference 0.77s" stat chips in the project deck.
        n_params = sum(p.numel() for p in self.net.parameters()) + \
            sum(p.numel() for p in self.trans.parameters())
        self.param_count = n_params
        self.param_count_str = f"{n_params / 1e6:.0f}M" if n_params >= 1e6 else f"{n_params / 1e3:.0f}K"

    @torch.no_grad()
    def generate_preview(self, prompt: str, motion_len: int = 90, time_steps: int = 20,
                          cond_scale: int = 2, topkr: float = 0.9) -> GeneratedMotion:
        """Fast path: runs the real model, returns smoothed (but not foot-IK
        corrected) joint positions directly in memory. No disk writes, no
        matplotlib render. This is what the Streamlit 'Generate' button uses."""
        tick = time.time()
        ids_length = torch.tensor([motion_len // 4]).long().to(self.device)
        text = [prompt]

        motion_ids = self.trans.generate(
            text, ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1,
        )
        half = motion_ids.shape[1] // 2
        ids1, ids2 = motion_ids[:, :half], motion_ids[:, half:]

        out1 = self.net.forward_decoder(ids1.unsqueeze(-1).to(self.device))
        out2 = self.net.forward_decoder(ids2.unsqueeze(-1).to(self.device))
        motion_output = torch.cat([out1, out2], dim=-1)

        motions_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], 2, -1)
        motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())
        motion = motions_output[0]  # (frames, 2, 262)

        def joints_for(person_idx):
            m = motion[:, person_idx]
            j3d = m[:, :22 * 3].reshape(-1, 22, 3)
            j3d = filters.gaussian_filter1d(j3d, 1, axis=0, mode="nearest")
            return j3d

        return GeneratedMotion(
            joints_a=joints_for(0), joints_b=joints_for(1), caption=prompt,
            fps=self.fps, gen_seconds=time.time() - tick,
        )

    def generate_and_save(self, prompt: str, out_name: str, vis_dir: str, npy_dir: str,
                           motion_len: int = 90, time_steps: int = 20, cond_scale: int = 2,
                           topkr: float = 0.9, foot_ik: bool = True):
        """Full-fidelity path: calls the repo's own, unmodified
        preprocess_plot_motion() -- same function infer.py itself calls --
        so saved .npy/.mp4 outputs match running infer.py directly (foot-IK
        corrected). Slower (includes the matplotlib .mp4 render); used by the
        'Save to disk' button, not for live preview."""
        from utils.plot_script import preprocess_plot_motion

        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)

        tick = time.time()
        ids_length = torch.tensor([motion_len // 4]).long().to(self.device)
        with torch.no_grad():
            motion_ids = self.trans.generate(
                [prompt], ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1,
            )
            half = motion_ids.shape[1] // 2
            ids1, ids2 = motion_ids[:, :half], motion_ids[:, half:]
            out1 = self.net.forward_decoder(ids1.unsqueeze(-1).to(self.device))
            out2 = self.net.forward_decoder(ids2.unsqueeze(-1).to(self.device))
            motion_output = torch.cat([out1, out2], dim=-1)
            motions_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], 2, -1)
            motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

        preprocess_plot_motion(motions_output[0], prompt, vis_dir, npy_dir, out_name, foot_ik=foot_ik)
        return time.time() - tick


def motion_to_viewer_json(gm: GeneratedMotion) -> dict:
    """Public wrapper -- used by app.py to hand a live GeneratedMotion to
    viewer_component.render_viewer() in the same dict shape a saved
    sample_clips/*.json file already uses."""
    return _motion_to_viewer_dict(gm)
