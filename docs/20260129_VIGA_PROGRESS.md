# VIGA Test Results

**Date:** January 30, 2026
**Purpose:** Testing the dynamic_scene pipeline with GPT-4o vision model for image-to-3D scene generation.

**Pipeline Mode: dynamic_scene**

## Pipeline Modes

VIGA supports multiple pipeline modes. This test used **dynamic_scene**:

| Mode | Purpose | Animation? | Use Case |
|------|---------|-----------|----------|
| **dynamic_scene** | Animated 3D scene from image + text | Yes | "Bottle tips over and rolls" |
| static_scene | Still 3D recreation of an image | No | Recreate a room from a photo |
| BlenderGym | Single-step scene editing | No | "Change the lamp color to red" |
| BlenderBench | Multi-step complex editing | No | Transform scene A to scene B |

> **Why dynamic_scene?** Both tests required animation -- objects moving, tipping, rolling, or smashing. The `dynamic_scene` pipeline generates keyframe animations in Blender.

## What are "Rounds"?

VIGA uses a **dual-agent iterative system**. Each **round** is one complete cycle:

**Generator** (Writes Blender code) --> **Blender** (Executes & renders) --> **Verifier** (Evaluates result) --> **Decision** (Good? Stop : Loop)

> **Example:** "Rounds: 15" means the agents went through 15 iterations of generate --> render --> evaluate --> refine until the Verifier was satisfied (or the Generator called `end`). More complex scenes = more rounds needed.

## Test 1: Green Tea Bottle

### Input

| Parameter | Value |
|-----------|-------|
| Image | Phone photo of Ito En green tea bottle |
| Description | "A green tea bottle sits on a desk. The bottle tips over and rolls across the surface." |

![Input Photo](test_results_images/01_greentea_input.jpg)
*Input Photo*

![Generated 3D Render](test_results_images/02_greentea_output.png)
*Generated 3D Render*

### Test Runs

| Run | Model | Status | Time | Notes |
|-----|-------|--------|------|-------|
| 1 | Llama 4 Scout (Groq) | **FAILED** | 11.29s | Text-only model, couldn't see image |
| 2 | GPT-4o | **SUCCESS** | 57.82s | 4 rounds, needed camera fix |
| 3 | GPT-4o (improved prompt) | **SUCCESS** | 86.86s | 6 rounds, animation keyframes |

### Generated Objects

- Bottle (green cylinder with transparency)
- Cap (white cylinder)
- Desk (brown plane)
- Camera + Point light
- Animation: tips over (frame 30), rolls (frame 60)

> **Output Files:**
> `D:\Projects\ProjectGenesis\VIGA\output\dynamic_scene\20260129_235234\greentea\blender_file_improved.blend`

## Test 2: Cezanne Still Life

### Input

| Parameter | Value |
|-----------|-------|
| Image | Cezanne painting - Still Life with Jug and Fruit |
| Description | "Throw a ball onto the table and smash all the objects on the table down." |

![Input Painting](test_results_images/03_artist_input.png)
*Input Painting*

![Generated 3D Render](test_results_images/04_artist_output.png)
*Generated 3D Render*

### Results

| Parameter | Value |
|-----------|-------|
| Model | GPT-4o |
| Status | **SUCCESS** |
| Time | 334.87 seconds (5.5 minutes) |
| Rounds | 15 |

### Generated Objects

- Apple (red/pink sphere)
- Pear1, Pear2 (elongated spheres)
- Cylinders (jug approximation)
- Multiple spheres (fruits)
- Planes (table, walls)
- 3 Cameras, Point light

> **Output File:**
> `D:\Projects\ProjectGenesis\VIGA\output\dynamic_scene\20260129_235727\artist\blender_file.blend`

## Issues Found & Fixes Applied

### 1. Camera/Lighting Issue

> **Problem:** AI sometimes forgets to add camera or lights, resulting in black renders.

**Fix Applied:** Updated prompt in `prompts/dynamic_scene/generator.py`:

```
[Guiding Principles]
- Start with geometry, camera, lighting; then add animation.
- CRITICAL: Every scene MUST include a camera.
- Set the camera as active: bpy.context.scene.camera = camera_object
- Position camera to frame all scene objects in view.
- Add at least one light source (POINT, SUN, or AREA).

[Quality Guidelines for Realistic Rendering]
- Use proper PBR materials with Principled BSDF shader
- For transparent objects: use Transmission Weight
- Add subdivision surface modifier for smooth curves
- Use multiple light sources: key, fill, rim
- Set render engine to CYCLES
```

### 2. API Key Issues

| Provider | Status | Notes |
|----------|--------|-------|
| Gemini | **Exhausted** | Free tier quota hit |
| Groq (Llama vision) | **Decommissioned** | Models removed |
| OpenAI GPT-4o | **Working** | Requires $5+ credits |

### 3. Blender 4.5 Compatibility

`Transmission` renamed to `Transmission Weight` in Principled BSDF shader.

## Current Limitations

- **Primitive-based:** Creates basic shapes (spheres, cylinders), not detailed meshes
- **No textures:** Materials are solid colors only
- **Camera positioning:** Often needs manual adjustment
- **Light energy:** Sometimes too dim

## Recommendations for Better Results

1. Use 3D asset generation APIs (Meshy, Rodin) for detailed objects
2. Pre-made asset libraries for common objects
3. HDRI environment lighting for realistic illumination
4. More specific prompts with exact dimensions
5. Iterative refinement with visual feedback loop

## Configuration

### API Keys Location

`D:\Projects\ProjectGenesis\VIGA\utils\_api_keys.py`

### Blender Path

`C:\Program Files\Blender Foundation\Blender 4.5\blender.exe`

### Run Command

```bash
cd D:/Projects/ProjectGenesis/VIGA
python runners/dynamic_scene.py --task <task_name> --model gpt-4o \
  --blender-command "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \
  --max-rounds 20
```

---

*Generated by Yuna | VIGA Progress Day 1*
*January 30, 2026 00:10 PST*
