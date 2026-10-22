# Changes from upstream (gohar-malik/intermask)

This is a copy of the public InterMask repo with Team 3's changes applied on
top. Starting point: a clean clone of `github.com/gohar-malik/intermask`.

## 1. CPU-compatibility patches (Month 1)

The official `environment.yml` pins `pytorch-cuda=11.7` and the `nvidia`
channel, and two files unconditionally assumed a GPU was present, which
crashed inference on a CPU-only machine (no GPU on our dev hardware).

- **`options/eval_option.py`** — `torch.cuda.set_device(opt.gpu_id)` ran
  unconditionally, crashing instantly with `--gpu_id -1`. Now only runs when
  a GPU is actually requested:
  ```python
  if opt.gpu_id != -1:
      torch.cuda.set_device(opt.gpu_id)
  ```
- **`models/vq/quantizer.py`** — the VQ-VAE codebook buffer was hardcoded
  `.cuda()`, crashing before the model could even load. Removed; `.to(device)`
  elsewhere already places it correctly:
  ```python
  self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))
  ```

Verified on our machine: `infer.py --gpu_id -1` runs end-to-end, 5/5 prompts,
zero errors (see Month 1 deck for the full terminal output).

**Note:** a few other unconditional `.cuda()` calls still exist elsewhere in
the repo (`visualization/InverseKinematics.py`, `data/quaternion.py`). These
were *not* touched, because the real CPU run above completed with zero
errors without patching them — meaning that code path isn't hit during
ordinary inference (`foot_ik=True`'s `remove_fs()` path doesn't exercise
`InverseKinematics.py`'s GPU branch). Flagging this here rather than silently
leaving it undocumented, in case a future change (e.g. BVH export) does hit
that code path and needs the same treatment.

## 2. 3D motion viewer (Month 2 addition)

New, additive only — nothing above was touched for this part, and nothing in
`infer.py`'s generation pipeline is modified or imported by it:

- **`export_motion_json.py`** — reads the `keypoint_npy` files `infer.py`
  already writes and converts one generated clip into a compact JSON file
  for the viewer.
- **`viewer/`** — a self-contained three.js page that renders both
  generated people as a real, orbit-able 3D skeleton (vendored locally, so
  it works with no internet connection — important for a live demo). See
  `README_3D_VIEWER.md` for exact run commands.

This replaces the paper's flat, fixed-camera stick-figure `.mp4` with
something the professor can rotate the camera around live — using joint
position data the pipeline already produces, no retraining or model changes
involved.
