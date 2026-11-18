"""
viewer_component.py

Renders the InterMask 3D motion viewer (the same orbit/zoom/pan, play/pause,
scrub viewer built in the earlier "3D viewer" milestone) inside a Streamlit
page, via streamlit.components.v1.html.

Why this file exists instead of just pointing an <iframe> at viewer/index.html:
Streamlit's component sandbox renders a single HTML string in an isolated
iframe -- it doesn't serve the rest of the repo (vendor/, sample_clips/)
alongside it as static files. So this module:
  1. reads the vendored three.js + OrbitControls.js from vendor/three/ (same
     files viewer/index.html already uses) and inlines them as base64
     `data:` URIs inside an import map, so the viewer still runs with ZERO
     internet connection, same guarantee as before;
  2. takes a motion dict already in memory (either loaded from a .json file,
     or produced live by inference_engine.py) and injects it directly as a
     JS object literal, instead of fetch()-ing a file -- there's no separate
     HTTP server in the Streamlit app, so this avoids needing one.

Nothing about the original viewer/index.html is changed; this is an additive
Streamlit-specific wrapper around the same rendering code.
"""
from __future__ import annotations  # Python 3.7-safe: postpones annotation evaluation

import base64
import html as html_lib
import json
from pathlib import Path

import streamlit.components.v1 as components

_HERE = Path(__file__).parent
_THREE_JS_PATH = _HERE / "vendor" / "three" / "build" / "three.module.min.js"
_ORBIT_CONTROLS_PATH = _HERE / "vendor" / "three" / "examples" / "jsm" / "controls" / "OrbitControls.js"


def _data_uri(path: Path, mime: str = "text/javascript") -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_html(motion: dict, accent_a: str, accent_b: str) -> str:
    three_uri = _data_uri(_THREE_JS_PATH)
    orbit_uri = _data_uri(_ORBIT_CONTROLS_PATH)
    motion_json = json.dumps(motion)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --bg: #0b0f14; --panel: #11161dcc; --border: #273040; --text: #e7edf5;
    --muted: #8b98ab; --accentA: {accent_a}; --accentB: {accent_b};
  }}
  html, body {{ margin: 0; padding: 0; height: 100%; background: var(--bg);
    overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  #canvas-wrap {{ position: fixed; inset: 0; }}
  canvas {{ display: block; }}
  .hud {{ position: fixed; z-index: 10; color: var(--text); }}
  #title-bar {{ top: 10px; left: 12px; right: 12px; display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
  #title-bar h1 {{ font-size: 13px; margin: 0; font-weight: 600; letter-spacing: 0.2px; }}
  #title-bar .caption {{ color: var(--muted); font-size: 11.5px; }}
  #legend {{ position: fixed; top: 10px; right: 12px; z-index: 10; display: flex; gap: 12px; font-size: 11px;
    color: var(--muted); background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 5px 9px; backdrop-filter: blur(6px); }}
  #legend span.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
  #bottom-bar {{ left: 12px; right: 12px; bottom: 10px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 12px; backdrop-filter: blur(6px); display: flex; align-items: center; gap: 10px; }}
  #bottom-bar button {{ background: #1b2330; color: var(--text); border: 1px solid var(--border); border-radius: 7px;
    width: 32px; height: 28px; cursor: pointer; font-size: 13px; }}
  #bottom-bar button:hover {{ background: #242e3e; }}
  #scrub {{ flex: 1; accent-color: var(--accentB); }}
  #frame-label {{ font-variant-numeric: tabular-nums; font-size: 11px; color: var(--muted); min-width: 80px; text-align: right; }}
  select {{ background: #1b2330; color: var(--text); border: 1px solid var(--border); border-radius: 7px; padding: 5px 7px; font-size: 11px; }}
  #error-banner {{ display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    z-index: 20; background: #2a1414; border: 1px solid #5a2a2a; color: #ffb4b4; padding: 12px 16px;
    border-radius: 10px; max-width: 480px; font-size: 12px; line-height: 1.5; }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div class="hud" id="title-bar">
  <h1>InterMask &middot; 3D Viewer</h1>
  <span class="caption" id="caption-text">Loading&hellip;</span>
</div>
<div id="legend">
  <span><span class="dot" style="background: var(--accentA);"></span>Person A</span>
  <span><span class="dot" style="background: var(--accentB);"></span>Person B</span>
</div>
<div class="hud" id="bottom-bar">
  <button id="btn-play" title="Play / Pause">&#9654;</button>
  <input id="scrub" type="range" min="0" max="0" value="0" step="1" />
  <span id="frame-label">0 / 0</span>
  <select id="speed">
    <option value="0.25">0.25x</option>
    <option value="0.5">0.5x</option>
    <option value="1" selected>1x</option>
    <option value="1.5">1.5x</option>
    <option value="2">2x</option>
  </select>
</div>
<div id="error-banner"></div>

<script type="importmap">
{{
  "imports": {{
    "three": "{three_uri}",
    "three/addons/controls/OrbitControls.js": "{orbit_uri}"
  }}
}}
</script>

<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

const MOTION_DATA = {motion_json};

function showError(msg) {{
  const el = document.getElementById("error-banner");
  el.textContent = msg;
  el.style.display = "block";
}}

const wrap = document.getElementById("canvas-wrap");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
scene.fog = new THREE.Fog(0x0b0f14, 6, 16);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(2.6, 1.8, 2.6);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
wrap.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x1a1a1a, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(3, 5, 2);
scene.add(key);

const grid = new THREE.GridHelper(10, 20, 0x2a3442, 0x1a212c);
scene.add(grid);

window.addEventListener("resize", () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function buildSkeleton(color) {{
  const group = new THREE.Group();
  const jointGeo = new THREE.SphereGeometry(0.028, 12, 12);
  const jointMat = new THREE.MeshStandardMaterial({{ color, roughness: 0.4, metalness: 0.1 }});
  const joints = [];
  for (let i = 0; i < 22; i++) {{
    const m = new THREE.Mesh(jointGeo, jointMat);
    group.add(m);
    joints.push(m);
  }}
  joints[15].scale.setScalar(1.6);
  const boneMat = new THREE.LineBasicMaterial({{ color, linewidth: 2 }});
  return {{ group, joints, boneMat, bones: [] }};
}}

function rebuildBones(skel, edges) {{
  for (const b of skel.bones) skel.group.remove(b.line);
  skel.bones = [];
  for (const [a, b] of edges) {{
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const line = new THREE.Line(geo, skel.boneMat);
    skel.group.add(line);
    skel.bones.push({{ line, a, b }});
  }}
}}

function updateSkeleton(skel, framePositions, heightOffset) {{
  for (let i = 0; i < 22; i++) {{
    const [x, y, z] = framePositions[i];
    skel.joints[i].position.set(x, y - heightOffset, z);
  }}
  for (const bone of skel.bones) {{
    const pa = skel.joints[bone.a].position;
    const pb = skel.joints[bone.b].position;
    const pos = bone.line.geometry.attributes.position;
    pos.setXYZ(0, pa.x, pa.y, pa.z);
    pos.setXYZ(1, pb.x, pb.y, pb.z);
    pos.needsUpdate = true;
  }}
}}

const skelA = buildSkeleton({json.dumps(accent_a)});
const skelB = buildSkeleton({json.dumps(accent_b)});
scene.add(skelA.group);
scene.add(skelB.group);

let motion = null;
let frame = 0;
let playing = true;
let speed = 1;
let acc = 0;

const scrub = document.getElementById("scrub");
const frameLabel = document.getElementById("frame-label");
const btnPlay = document.getElementById("btn-play");
const speedSel = document.getElementById("speed");
const captionText = document.getElementById("caption-text");

btnPlay.addEventListener("click", () => {{
  playing = !playing;
  btnPlay.innerHTML = playing ? "&#9654;" : "&#10073;&#10073;";
}});
speedSel.addEventListener("change", (e) => {{ speed = parseFloat(e.target.value); }});
scrub.addEventListener("input", (e) => {{
  frame = parseInt(e.target.value, 10);
  playing = false;
  btnPlay.innerHTML = "&#9654;";
  renderFrame();
}});

function heightOffsetFor(frames) {{
  let minY = Infinity;
  for (const f of frames) for (const [, y] of f) if (y < minY) minY = y;
  return minY;
}}

function renderFrame() {{
  if (!motion) return;
  updateSkeleton(skelA, motion.personA[frame], motion._offsetA);
  updateSkeleton(skelB, motion.personB[frame], motion._offsetB);
  scrub.value = frame;
  frameLabel.textContent = `${{frame + 1}} / ${{motion.frames}}`;
}}

function loadMotion(data) {{
  if (!data.personA || !data.personB || !data.edges) {{
    throw new Error("motion data is missing personA / personB / edges");
  }}
  data._offsetA = heightOffsetFor(data.personA);
  data._offsetB = heightOffsetFor(data.personB);
  rebuildBones(skelA, data.edges);
  rebuildBones(skelB, data.edges);
  motion = data;
  frame = 0;
  scrub.max = data.frames - 1;
  captionText.textContent = data.caption ? `"${{data.caption}}"` : "(no caption)";
  renderFrame();
}}

try {{
  loadMotion(MOTION_DATA);
}} catch (err) {{
  console.error(err);
  showError("Couldn't load motion data: " + err.message);
}}

let last = performance.now();
function tick(now) {{
  const dt = (now - last) / 1000;
  last = now;
  if (motion && playing) {{
    acc += dt * motion.fps * speed;
    while (acc >= 1) {{
      frame = (frame + 1) % motion.frames;
      acc -= 1;
    }}
    renderFrame();
  }}
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}}
requestAnimationFrame(tick);
</script>
</body>
</html>"""


def render_viewer(motion: dict, height: int = 560, accent_a: str = "#ff8a3d", accent_b: str = "#3da9ff", key: str | None = None):
    """
    Renders one InterMask motion clip as an interactive 3D viewer in the
    current Streamlit page.

    motion: dict with keys caption, fps, frames, jointNames, edges, personA,
            personB -- the same schema export_motion_json.py writes.
    """
    html = _build_html(motion, accent_a, accent_b)
    components.html(html, height=height, scrolling=False)


def load_motion_json(path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
