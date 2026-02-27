# VIGA Test Results - January 30, 2026

## Summary
Testing the dynamic_scene pipeline with GPT-4o vision model for image-to-3D scene generation.

---

## Test 1: Green Tea Bottle (greentea)

### Input
- **Image:** `C:\Users\kingy\Downloads\Phone Link\PXL_20260129_082438961.jpg`
- **Description:** "A green tea bottle sits on a desk. The bottle tips over and rolls across the surface."

### Results

#### Run 1 (Llama 4 Scout - Groq) - FAILED
- **Model:** meta-llama/llama-4-scout-17b-16e-instruct
- **Issue:** Text-only model, couldn't see the image
- **Output:** Just flat planes labeled "Desk"
- **Time:** 11.29 seconds

#### Run 2 (GPT-4o) - SUCCESS
- **Model:** gpt-4o
- **Time:** 57.82 seconds
- **Rounds:** 4
- **Objects created:** Camera, Cylinder (bottle), Cylinder (cap), Plane (desk), Point (light)
- **Issue:** Camera not framing objects, black render initially
- **After manual fix:** Green bottle with white cap visible

#### Run 3 (GPT-4o with improved prompt) - BETTER
- **Model:** gpt-4o  
- **Time:** 86.86 seconds
- **Rounds:** 6
- **Objects created:** Bottle, Cap, Desk (properly named), with materials
- **Features:** Animation keyframes (tips over and rolls)
- **After material fix:** Green translucent bottle, white cap, brown desk

### Output Files
- Original: `D:\Projects\ProjectGenesis\VIGA\output\dynamic_scene\20260129_234713\greentea\blender_file.blend`
- Improved: `D:\Projects\ProjectGenesis\VIGA\output\dynamic_scene\20260129_235234\greentea\blender_file_improved.blend`

### Render (after fixes)
- Green cylinder bottle with white cap on brown desk
- Basic primitives, not photorealistic

---

## Test 2: Cézanne Still Life (artist)

### Input
- **Image:** `D:\Projects\ProjectGenesis\VIGA\data\dynamic_scene\artist\target.png`
- **Description:** "Throw a ball onto the table and smash all the objects on the table down."
- **Source:** Cézanne painting - still life with jug and fruits

### Results (GPT-4o)
- **Model:** gpt-4o
- **Time:** 334.87 seconds (5.5 minutes)
- **Rounds:** 15
- **Objects created:**
  - Apple (sphere)
  - Pear1, Pear2 (elongated spheres)
  - Cylinder, Cylinder.001 (jug)
  - Multiple spheres (fruits)
  - Planes (table, walls)
  - 3 Cameras
  - Point light

### Output Files
- `D:\Projects\ProjectGenesis\VIGA\output\dynamic_scene\20260129_235727\artist\blender_file.blend`

### Render
- Red/pink sphere (apple)
- Beige pear shape
- Gray table and walls
- Basic primitive approximation of the painting

---

## Issues Found & Fixed

### 1. Camera/Lighting Issue (FIXED)
**Problem:** AI sometimes forgets to add camera or lights, resulting in black renders.

**Fix Applied:** Updated prompt in `prompts/dynamic_scene/generator.py`:
```
[Guiding Principles]
• Start with geometry, camera, lighting; then add animation.
• CRITICAL: Every scene MUST include a camera. Without a camera, no image can be rendered.
• Set the camera as the active scene camera: bpy.context.scene.camera = camera_object
• Position the camera to frame all scene objects in view.
• Add at least one light source (POINT, SUN, or AREA) so objects are visible.

[Quality Guidelines for Realistic Rendering]
• Use proper PBR materials with nodes (Principled BSDF shader)
• For transparent objects: use Principled BSDF with Transmission Weight
• Add subdivision surface modifier for smooth curved objects
• Use multiple light sources: key light, fill light, rim/back light
• Set render engine to CYCLES for photorealistic results
• Add environment/background
• Match proportions from reference image
```

### 2. API Key Issues
- Gemini free tier quota exhausted
- OpenAI requires credits ($5 minimum)
- Groq vision models (llama-3.2-90b/11b) decommissioned
- **Working:** GPT-4o with paid OpenAI credits

### 3. Blender 4.5 Compatibility
- `Transmission` renamed to `Transmission Weight` in Principled BSDF

---

## Limitations

1. **Primitive-based generation:** Creates basic shapes (spheres, cylinders, planes), not detailed meshes
2. **No texture generation:** Materials are solid colors, no image textures
3. **Camera positioning:** Often needs manual adjustment after generation
4. **Light energy:** Sometimes too dim, needs boosting

## Recommendations for Better Results

1. **Use 3D asset generation APIs** (Meshy, Rodin) for detailed objects
2. **Pre-made asset libraries** for common objects
3. **HDRI environment lighting** for realistic illumination
4. **More specific prompts** with exact dimensions and positions
5. **Iterative refinement** with visual feedback loop

---

## Configuration

### API Keys Location
`D:\Projects\ProjectGenesis\VIGA\utils\_api_keys.py`

### Blender Path
`C:\Program Files\Blender Foundation\Blender 4.5\blender.exe`

### Run Command
```powershell
cd D:/Projects/ProjectGenesis/VIGA
python runners/dynamic_scene.py --task <task_name> --model gpt-4o --blender-command "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" --max-rounds 20
```

---

*Generated: 2026-01-30 00:05 PST*
