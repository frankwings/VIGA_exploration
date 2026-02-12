# VIGA Pipeline Revisited

**Date:** 2026-02-12
**Author:** kingyy (win/vscode/opus/hum)

How VIGA reconstructs visual content through iterative code generation — with SAM3D reconstruction, Meshy text-to-3D, and Blender rendering working together in a closed-loop pipeline.

---

## 1. High-Level Pipeline

```
                         ┌──────────────────────┐
                         │    Target Image       │
                         │  (photograph / art)   │
                         └───────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
   ┌────────────────────┐ ┌──────────────────┐  ┌────────────────────┐
   │      SAM3D         │ │   Description    │  │    Meshy Cache     │
   │  (image -> 3D)     │ │  (text prompt)   │  │ (previous assets/) │
   └─────────┬──────────┘ └────────┬─────────┘  └─────────┬──────────┘
             │                     │                       │
             │  GLBs + transforms  │  task description     │  cached GLBs
             ▼                     ▼                       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Generator Agent (VLM)                       │
   │  Writes Blender Python code, calls tools, iterates with        │
   │  verifier feedback. Chooses best GLB per object (SAM3D/Meshy). │
   └──────────────┬──────────────────────────────┬──────────────────┘
                  │ Blender Python script         │ get_better_object()
                  ▼                               ▼
   ┌────────────────────┐              ┌────────────────────┐
   │  Blender Executor  │              │   Meshy API/Cache  │
   │  (EEVEE render)    │              │   (text -> 3D)     │
   └─────────┬──────────┘              └─────────┬──────────┘
             │ rendered PNGs                      │ new GLB file
             ▼                                    │
   ┌────────────────────┐                         │
   │  Verifier Agent    │◀────────────────────────┘
   │  (VLM)             │
   │  Compares render   │
   │  to target image   │
   └─────────┬──────────┘
             │ structured feedback
             ▼
        Back to Generator
        (next round)
```

---

## 2. Module Breakdown

### 2.1 SAM3D — Image-to-3D Reconstruction

**Purpose:** Reconstruct 3D objects directly from the target photograph. SAM segments the image into per-object masks, then SAM3D (built on TRELLIS) lifts each mask into a 3D mesh with vertex colors.

**MCP Tool Server:** `tools/sam3d/init.py`

**Input:**
- `target.png` — the target photograph
- Blender command + file paths

**Processing (2 stages):**

| Stage | Process | Tool |
|-------|---------|------|
| 1. Segmentation | SAM (ViT-H) detects all objects, produces binary masks | `sam_worker.py` |
| 2. Reconstruction | SAM3D reconstructs each mask into a 3D mesh (GLB) | `sam3d_worker.py` (per-object subprocess) |

**Output:**
```
{output_dir}/sam_init/
├── all_masks.npy                  # Array of binary segmentation masks
├── all_masks_object_names.json    # Human-readable object names (fuzzy-matched)
├── ceramic_jug.glb                # Per-object 3D mesh with vertex colors
├── ceramic_jug.json               # Per-object transform metadata
├── green_apple.glb
├── green_apple.json
├── ...
├── object_transforms.json         # Consolidated transforms for all objects
│     {
│       "ceramic_jug": {
│         "translation": [0.581, 0.089, 1.540],
│         "rotation": [0, 0, 0, 1],      // quaternion (xyzw)
│         "scale": 0.571
│       }, ...
│     }
└── scene.blend                    # All GLBs imported into a single Blender file
```

**Characteristics:**
- Runs on GPU (RTX 5080: ~2 hours for 8 objects at 1024px)
- Captures scene layout and relative positions from the photo
- Geometry quality varies: simple shapes (fruits, bottles) reconstruct well; complex shapes (keyboards, headphones) are often blobs
- Vertex colors only (no UV textures)

---

### 2.2 Meshy — Text-to-3D Generation

**Purpose:** Generate 3D assets from text descriptions. Used to replace poor-quality SAM3D reconstructions or to generate objects from scratch when no target image is available.

**MCP Tool Server:** `tools/assets/meshy.py`
**API Client:** `tools/assets/meshy_api.py`

**Input (per object):**
- `object_name` — short name (e.g., "green tea bottle")
- `object_description` — text prompt (e.g., "Ito En green tea PET bottle, 500ml")
- `reference_type` — "text" (text-to-3D) or "image" (crops target image first)

**Processing (with cache check):**

```
get_better_object("keyboard", "RGB mechanical keyboard")
        │
        ▼
  ┌─────────────────────┐
  │  Check local cache   │  Fuzzy-match against data/{task}/assets/*.glb
  │  (previous_assets/)  │  Normalizes: lowercase, remove spaces, handle plurals
  └──────────┬──────────┘
             │
      ┌──────┴──────┐
      │ cache hit   │ cache miss + API key available
      ▼             ▼
  Return path   ┌─────────────────────┐
  (instant)     │  Meshy API Pipeline  │
                │  1. Create preview   │  (~2 min)
                │  2. Refine to HD     │  (~5 min)
                │  3. Download GLB     │
                └──────────┬──────────┘
                           │
                           ▼
                    Save to assets/keyboard.glb
```

**Output:**
```
data/{mode}/{task}/assets/
├── green_tea_bottle.glb      # High-quality textured 3D mesh
├── keyboard.glb
├── headphones.glb
├── table.glb
├── envelope.glb
└── meshy.log                 # API call log with task IDs and timing
```

**Characteristics:**
- Local cache hit is instant (0 API calls); cache miss takes ~7 min per object via API
- Produces UV-textured meshes (much higher visual quality than SAM3D vertex colors)
- Fuzzy name matching handles variations: "jug" matches "ceramic_jug.glb", "pear" matches "bartlett_pear.glb"
- API key is optional — local-only mode works for cached assets

---

### 2.3 Generator Agent — Code Synthesis

**Purpose:** Write Blender Python scripts that import GLBs, arrange objects, set up lighting/camera, and render the scene. Iterates based on Verifier feedback.

**Implementation:** `agents/generator.py`
**VLM:** GPT-5 (or GPT-4o, Claude, Gemini — configurable)

**Input:**
- System prompt (from `prompts/{mode}/generator.py`)
- Target image + text description
- SAM3D results (injected into memory if available)
- Verifier feedback (from previous rounds)

**Processing (per round):**

```
Round N:
  1. VLM generates a tool call:
     ├── initialize_plan(thought, plan)        # Round 0 only
     ├── get_better_object(name, description)  # Fetch/generate GLB via Meshy
     ├── execute_and_evaluate(thought, code)   # Run Blender script + render
     └── end(thought)                          # Signal completion

  2. Tool executes and returns result

  3. If execute_and_evaluate returned images:
     → Verifier Agent runs (compares render to target)
     → Feedback injected into Generator memory

  4. Generator receives tool result + verifier feedback
     → Generates next tool call (Round N+1)
```

**SAM3D Results Injection:**
When `--sam3d-results` is provided (or SAM3D runs automatically), the Generator receives context like:

```
SAM3D reconstructed 8 objects from the target image:
- ceramic_jug: D:/Projects/.../sam_init/ceramic_jug.glb
  translation=[0.581, 0.089, 1.540] rotation=[0,0,0,1] scale=0.571
- green_apple: D:/Projects/.../sam_init/green_apple.glb
  translation=[0.341, -0.161, 1.365] rotation=[0,0,0,1] scale=0.202
...
```

This is injected into `memory[1].content` so the VLM knows all available GLB paths and transforms.

**Output (per round):**
```
output/{mode}/{timestamp}/{task}/
├── scripts/
│   ├── 1.py              # Blender Python script (complete, self-contained)
│   ├── 2.py
│   └── ...
├── renders/
│   ├── 1/
│   │   ├── Camera.png        # Static scene: single render
│   │   └── state.blend       # Blender state snapshot (for undo)
│   ├── 2/
│   │   ├── Camera_f0001.png  # Dynamic scene: keyframe renders
│   │   ├── Camera_f0090.png
│   │   ├── Camera_f0180.png
│   │   └── state.blend
│   └── ...
├── generator_memory.json     # Full conversation history (~10-15 MB)
└── blender_file.blend        # Final accumulated Blender scene
```

---

### 2.4 Blender Executor — Script Execution & Rendering

**Purpose:** Execute Generator-written Blender Python scripts in headless mode and capture rendered images.

**MCP Tool Server:** `tools/blender/exec.py`

**Input:**
- Complete Blender Python script (from Generator)
- Blender wrapper script (e.g., `generator_script_eevee.py` — sets EEVEE engine)

**Processing:**

```
Generator calls execute_and_evaluate(code="import bpy; ...")
        │
        ▼
  Save script to scripts/{N}.py
        │
        ▼
  Run Blender in background:
    blender --background scene.blend --python wrapper.py -- {script} {render_dir}
        │
        ▼
  Wrapper script:
    1. Sets render engine (EEVEE / Cycles)
    2. Executes Generator's code (which imports GLBs, sets camera, etc.)
    3. Renders to {render_dir}/Camera.png (or Camera_f{NNNN}.png for dynamic)
    4. Saves state to state.blend
        │
        ▼
  Collect rendered PNGs → return to Generator
  If images found → trigger Verifier
```

**Output:**
- Rendered PNG images (resolution set by wrapper script, typically 512x512 or 1024x1024)
- `state.blend` snapshot per round (enables undo)
- stdout/stderr captured to temp files (Windows pipe deadlock avoidance)

**Key Implementation Details:**
- All paths resolved to absolute with `.resolve()` — Blender's working directory differs from the project root
- Windows: subprocess commands quoted for paths with spaces
- Render engine: `BLENDER_EEVEE_NEXT` (Blender 4.x EEVEE) for speed; Cycles available for quality

---

### 2.5 Verifier Agent — Visual Feedback

**Purpose:** Compare rendered output to the target image and provide structured feedback for the Generator to improve the scene.

**Implementation:** `agents/verifier.py`
**VLM:** Same as Generator (GPT-5 by default)

**Input:**
- Rendered images from current round
- Target image
- Scene metadata (object list, camera position)

**Processing:**

```
Verifier receives render + target:
  1. Analyze composition: object placement, scale, proportions
  2. Analyze appearance: colors, materials, textures, lighting
  3. Identify missing/extra objects
  4. Optionally use tools:
     ├── investigate("rotate camera left")  # Explore the scene
     ├── set_camera(x, y, z, rx, ry, rz)   # Move to specific viewpoint
     ├── set_keyframe(frame)                # Check animation frames
     └── get_scene_info()                   # Read object transforms
  5. Call end(feedback_text)
```

**Output:**
- Structured text feedback (e.g., "The jug is too small and positioned too far left. The lighting is too flat — add a key light from the upper left. The green apples need more saturated color.")
- Saved to `verifier_memory.json` (~10-15 MB)

**Characteristics:**
- Memory can be cleared each round (`clear_memory=True`) or accumulated
- Investigator tools let the Verifier inspect the 3D scene from multiple angles before giving feedback

---

## 3. Pipeline Modes

### 3.1 Static Scene with Meshy Only (`--prompt-setting=get_asset`)

```
Target Image + Description
        │
        ▼
  Phase 1: Asset Acquisition (max 5 rounds)
    Generator calls get_better_object() for each object
    Meshy checks cache → API fallback → returns GLB path
        │
        ▼
  Phase 2: Scene Composition (remaining rounds)
    Generator writes Blender Python:
      - Import all GLBs (exact absolute paths)
      - Position, scale, rotate to match target
      - Add ground plane, lighting, camera
      - Render
    Verifier provides feedback → Generator refines
        │
        ▼
  Output: scripts/, renders/, assets/*.glb
```

**Best for:** Scenes where SAM3D is unavailable or all objects need high-quality textured meshes.

### 3.2 Static/Dynamic Scene with SAM3D + Meshy (`--prompt-setting=get_asset_sam3d`)

```
Target Image + Description
        │
        ▼
  Stage 0: SAM3D Reconstruction (automatic)
    SAM segments target → SAM3D reconstructs per-object GLBs
    Results injected into Generator memory
        │
        ▼
  Phase 1: Evaluate & Replace (max 5 rounds)
    Generator imports SAM3D GLBs → renders → evaluates quality
    For POOR objects: call get_better_object() (Meshy)
    For GOOD objects: keep SAM3D GLB as-is
        │
        ▼
  Phase 2: Scene Composition (remaining rounds)
    Generator writes Blender Python using BEST GLB per object:
      - SAM3D GLB (if reconstruction was good)
      - Meshy GLB (if SAM3D was poor, replaced in Phase 1)
    Verifier feedback loop
        │
        ▼
  [Dynamic scenes only] Phase 3: Animation
    Generator adds rigid body physics:
      - Table: PASSIVE rigid body
      - Objects: ACTIVE rigid bodies
      - Ball: keyframed initial velocity
      - Bake simulation (180-250 frames)
    Render keyframes (f0001, f0090, f0180)
        │
        ▼
  Output: scripts/, renders/, sam_init/*.glb, assets/*.glb
```

**Best for:** Scenes with a target photograph where SAM3D can capture layout and some objects reconstruct well (fruits, bottles, simple shapes). Complex objects (keyboards, headphones) get replaced by Meshy.

### 3.3 Pre-computed SAM3D (`--sam3d-results=path/to/sam_init`)

Same as 3.2, but skips the SAM3D reconstruction stage entirely. Loads pre-existing GLBs and transforms from a previous run's `sam_init/` directory. This saves ~2 hours of GPU time when iterating on scene composition.

---

## 4. What Each Module Produces

| Module | Input | Output | Format | Typical Size |
|--------|-------|--------|--------|-------------|
| **SAM** | target.png | Per-object binary masks | `.npy` array | ~1 MB |
| **SAM3D** | mask + cropped image | 3D mesh with vertex colors | `.glb` + `.json` | 5-15 MB per object |
| **Meshy (cache)** | object name | Textured 3D mesh | `.glb` | 2-10 MB per object |
| **Meshy (API)** | text prompt | Textured 3D mesh (HD) | `.glb` | 2-10 MB per object |
| **Generator** | prompt + context | Blender Python script | `.py` | 2-5 KB per round |
| **Executor** | script + .blend | Rendered images + state | `.png` + `.blend` | 0.5-2 MB per image |
| **Verifier** | render + target | Text feedback | string | ~500 chars |

---

## 5. SAM3D vs Meshy: When Each is Used

| Aspect | SAM3D | Meshy |
|--------|-------|-------|
| **Source** | Target photograph | Text description |
| **Geometry** | Reconstructed from image | Generated from scratch |
| **Textures** | Vertex colors only | Full UV-mapped textures |
| **Scene layout** | Captures real positions/scales | No spatial information |
| **Quality** | Varies: good for simple shapes, poor for complex | Consistently high |
| **Speed** | ~15 min/object (GPU) | ~7 min/object (API) or instant (cache) |
| **Best for** | Fruits, bottles, simple objects | Keyboards, headphones, furniture |
| **Failure mode** | Blobs, missing details | Wrong interpretation of prompt |

**In practice (green tea desktop scene):**
- SAM3D kept: green tea bottle (good geometry + vertex colors)
- Meshy replaced: keyboard (SAM3D blob), headphones (SAM3D blob), table, envelope

**In practice (Cezanne still life scene):**
- SAM3D kept: 5 fruits — green apples + pears (good enough shapes)
- Meshy replaced: ceramic jug, white plate, one pear (poor SAM3D quality)

---

## 6. Memory & Context Management

VIGA uses a **sliding window** to manage VLM conversation context:

- System prompt: target image + task description + SAM3D paths (if available)
- Window size: L = 12-24 most recent rounds
- Each round adds: assistant message (tool call) + tool result (images + text) + verifier feedback
- Reduces token cost from O(N^2) to O(N*k) over long runs
- Full history saved to `generator_memory.json` / `verifier_memory.json` for debugging

---

## 7. Output Structure

### Static Scene
```
output/static_scene/{timestamp}/{task}/
├── scripts/
│   ├── 1.py, 2.py, ..., N.py         # One Blender Python script per round
├── renders/
│   ├── 1/
│   │   ├── Camera.png                 # Rendered image
│   │   └── state.blend                # Scene snapshot (for undo)
│   ├── 2/ ...
├── sam_init/                          # (if SAM3D was used)
│   ├── *.glb, *.json                  # Per-object reconstructions
│   └── object_transforms.json         # Consolidated pose data
├── generator_memory.json              # ~10-15 MB
├── verifier_memory.json               # ~10-15 MB
└── blender_file.blend                 # Accumulated final scene

data/static_scene/{task}/assets/       # Meshy cache (persists across runs)
├── green_tea_bottle.glb
├── keyboard.glb
└── meshy.log
```

### Dynamic Scene
```
output/dynamic_scene/{timestamp}/{task}/
├── scripts/
│   ├── 1.py, 2.py, ..., N.py
├── renders/
│   ├── 1/
│   │   ├── Camera_f0001.png           # Start frame
│   │   ├── Camera_f0090.png           # Mid frame
│   │   ├── Camera_f0180.png           # End frame
│   │   └── state.blend
│   ├── 2/ ...
├── sam_init/                          # (if SAM3D was used)
├── generator_memory.json
├── verifier_memory.json
└── blender_file.blend

data/dynamic_scene/{task}/assets/      # Meshy cache
├── jug.glb
├── plate.glb
└── meshy.log
```

---

## 8. Agent Tools Reference

### Generator Tools

| Tool | Server | Purpose | Output |
|------|--------|---------|--------|
| `initialize_plan` | `tools/initialize_plan.py` | Write high-level plan before acting | Plan text |
| `execute_and_evaluate` | `tools/blender/exec.py` | Run Blender Python + render | PNGs + state.blend |
| `get_better_object` | `tools/assets/meshy.py` | Fetch/generate 3D asset | GLB file path |
| `get_scene_info` | `tools/blender/exec.py` | Query scene objects, transforms | JSON text |
| `undo_last_step` | `tools/blender/exec.py` | Restore previous state.blend | Confirmation |
| `reconstruct_full_scene` | `tools/sam3d/init.py` | SAM3D full pipeline (auto-called) | GLB paths + transforms |
| `end` | built-in | Signal task completion | — |

### Verifier Tools

| Tool | Server | Purpose | Output |
|------|--------|---------|--------|
| `initialize_viewpoint` | `tools/blender/investigator.py` | Compute diagnostic camera views | View list |
| `set_camera` | `tools/blender/investigator.py` | Move camera to specific pose | Rendered PNG |
| `investigate` | `tools/blender/investigator.py` | Natural language camera control | Rendered PNG |
| `set_keyframe` | `tools/blender/investigator.py` | Navigate animation timeline | Rendered PNG |
| `set_visibility` | `tools/blender/investigator.py` | Toggle object visibility | Confirmation |
| `get_scene_info` | `tools/blender/investigator.py` | Scene state summary | JSON text |
| `end` | built-in | Return feedback to Generator | Feedback text |

---

## 9. Supported Models

VIGA works with multiple vision-language models as the Generator and Verifier:

- **OpenAI**: gpt-5, gpt-4o, gpt-4-turbo, gpt-4o-mini
- **Anthropic**: claude-sonnet-4, claude-opus-4.5
- **Google**: gemini-2.5-pro, gemini-2.0-flash
- **Qwen**: qwen-vl-max, qwen-vl-plus

---

## 10. Run Commands

### Static scene (Meshy only)
```bash
python runners/static_scene.py \
  --task=greentea --model=gpt-5 \
  --blender-command="C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \
  --blender-script="data/static_scene/generator_script_eevee.py" \
  --prompt-setting=get_asset \
  --max-rounds=25
```

### Static scene (SAM3D + Meshy)
```bash
python runners/static_scene.py \
  --task=greentea --model=gpt-5 \
  --blender-command="C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \
  --blender-script="data/static_scene/generator_script_eevee.py" \
  --prompt-setting=get_asset_sam3d \
  --generator-tools="tools/blender/exec.py,tools/generator_base.py,tools/assets/meshy.py,tools/sam3d/init.py,tools/initialize_plan.py" \
  --max-rounds=25
```

### Dynamic scene (SAM3D + Meshy, pre-computed)
```bash
python runners/dynamic_scene.py \
  --task=artist --model=gpt-5 \
  --blender-command="C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \
  --prompt-setting=get_asset_sam3d \
  --sam3d-results="output/dynamic_scene/20260210_194152/artist/sam_init" \
  --max-rounds=25
```

---

*Generated by VIGA (Vision-as-Inverse-Graphics Agent)*
*Analysis by Claude Opus 4.6*
