# VIGA Workflow

*Vision-Guided Agentic 3D Scene Generation*

## Core Components & Third-Party Tools

### Vision Language Models (Choose One)

**OpenAI GPT-4o**
Primary vision model. Analyzes images and generates Blender Python code.
- Best image understanding
- Reliable code generation
- Production ready
- `api.openai.com/v1/chat/completions`

**Google Gemini**
Alternative vision model. Google multimodal AI with vision capabilities.
- Multimodal vision
- Good vision capabilities
- Fast responses
- `generativelanguage.googleapis.com`

**Anthropic Claude**
Claude models with vision. Good for complex reasoning.
- Strong reasoning
- Detailed outputs
- Enterprise ready
- `api.anthropic.com/v1`

**Groq (Llama)**
Fast inference for Llama models. High-speed inference platform.
- Ultra-fast inference
- High throughput
- Vision models deprecated
- `api.groq.com/openai/v1`

**Qwen (Local)**
Self-hosted Qwen models. Privacy-focused, runs on your hardware.
- Run locally
- Data stays local
- Requires GPU
- `localhost:8000/v1`

### 3D Generation & Rendering

**Blender**
Open-source 3D creation software. Executes Python scripts for geometry, materials, lighting, animation.
- Cycles/EEVEE rendering
- Python scripting API
- Procedural modeling
- `C:\Program Files\Blender Foundation\Blender 4.5\`

**Meshy API**
Text-to-3D and Image-to-3D. Creates detailed 3D assets with PBR textures.
- Text-to-3D generation
- Image-to-3D reconstruction
- GLB/FBX export
- `tools/assets/meshy.py`

**Infinigen**
Procedural generator for realistic 3D worlds. Nature scenes, indoor environments.
- Procedural terrain & plants
- Indoor scene generation
- Asset libraries
- `utils/third_party/infinigen/`

### Segmentation & 3D Reconstruction

**SAM (Segment Anything)**
Meta's automatic segmentation. Detects all objects in an image.
- Auto-segment everything
- High-quality masks
- ViT-H backbone
- `utils/third_party/sam/`

**SAM3 (with Text Prompts)**
SAM version 3. Segment specific objects using text descriptions.
- Text-prompted segmentation
- "Segment the bottle"
- More precise control
- `utils/third_party/sam3/`

**SAM3D (2D to 3D)**
Reconstructs 3D meshes from segmented 2D images. Exports to GLB.
- Mask to 3D mesh
- Depth estimation
- GLB export
- `utils/third_party/sam3d/`

### Additional Tools

**Slides (PPTX)**
PowerPoint slide generation using Python-pptx and unoconv for rendering.
- Auto-generate presentations
- PPTX to PNG screenshots
- unoconv conversion
- `tools/slides/`

**MCP (Model Context Protocol)**
Tool server framework. All tools run as MCP servers with JSON-RPC.
- FastMCP framework
- Tool registration
- Async execution
- `mcp.server.fastmcp`

**PIL/Pillow**
Image processing library. Resizes, converts, and encodes images for API calls.
- Image resizing
- Base64 encoding
- Format conversion
- `Python: PIL/Pillow`

## Pipeline Modes Explained

### dynamic_scene -- Animated 3D Scene Generation

2D image + text description --> Full 3D scene with animation and physics

**Flow:** Image + Text ("Bottle tips over and rolls") --> GPT-4o + Blender (Vision analysis to Python code) --> Animated Scene (.blend + render frames)

**Features:**
- Keyframe animation generation
- Physics simulation support
- Multi-object interactions

**Tools Used:**
- **GPT-4o** -- Required
- **Blender** -- Required
- **Meshy/SAM** -- Optional

**Example:** Green tea bottle tips over on desk

### static_scene -- Static 3D Scene Reconstruction

2D image --> Still 3D composition matching the photo

**Flow:** Single Image (Photo of a scene) --> GPT-4o + Meshy (Generate detailed 3D assets) --> 3D Scene (Accurate recreation)

**Features:**
- High-fidelity asset generation
- PBR material recreation
- Accurate lighting match

**Tools Used:**
- **GPT-4o** -- Required
- **Meshy** -- Default
- **Blender** -- Required

**Example:** Recreate a room interior from photo

### BlenderGym -- Single-Step 3D Editing

Existing Blender scene + instruction --> One specific modification

**Flow:** .blend + Instruction ("Change color to red") --> GPT-4o Edit (Generate edit script) --> Modified Scene (Single change applied)

**Features:**
- Simple, focused modifications
- Uses Infinigen procedural assets
- Fast single-step edits

**Tools Used:**
- **GPT-4o** -- Required
- **Infinigen** -- Used
- **Blender** -- Required

**Example:** "Move the lamp to the left"

### BlenderBench -- Multi-Step Complex Editing

Benchmark with Level 1-3 difficulty -- Multiple sequential edits

**Flow:** Before + Target (Two scene images) --> Iterative Editing (Multi-step transforms) --> Matched Target (Scene B achieved)

**Difficulty Levels:**
- **Level 1:** Easy (1-2 edits)
- **Level 2:** Medium (3-5 edits)
- **Level 3:** Hard (6+ edits)

**Tools Used:**
- **GPT-4o** -- Required
- **Infinigen** -- Used
- **Blender** -- Required

**Example:** Transform scene A into scene B

### SlideBench -- PowerPoint Slide Generation

Text descriptions --> Professional presentation slides (2D, not 3D)

**Flow:** Description (Slide content text) --> python-pptx (Generate layout) --> .pptx File (Ready presentation)

**Features:**
- Auto layout generation
- Text, images, charts
- PNG export via unoconv

**Tools Used:**
- **GPT-4o** -- Required
- **python-pptx** -- Required
- **unoconv** -- For export

**Example:** "Create a title slide about AI"

### Tool Usage by Mode

| Mode | GPT-4o | Meshy | SAM/SAM3D | Infinigen | Blender |
|------|--------|-------|-----------|-----------|---------|
| **dynamic_scene** | Required | Optional | Optional | -- | Required |
| **static_scene** | Required | Default | Optional | Optional | Required |
| **BlenderGym** | Required | -- | -- | Used | Required |
| **BlenderBench** | Required | -- | -- | Used | Required |
| **SlideBench** | Required | -- | -- | -- | -- |

### Where Each API is Used

| Component | What It Does | When Called |
|-----------|-------------|------------|
| **GPT-4o (ChatGPT)** | Analyzes image --> Plans scene --> Generates Blender Python code | Every round of Generator Agent (agents/generator.py) |
| **SAM / SAM3D** | Auto-segments ALL objects --> Reconstructs 3D meshes --> Creates initial scene | **Stage 2 (Initialization):** `--prompt-setting init` -- Runs BEFORE agent loop starts |
| **Meshy API** | Text/Image-to-3D: Creates detailed 3D models on-demand | **Stage 3+ (During Loop):** When agent calls `get_better_object` -- Upgrades individual objects during iterations |

> **SAM3D vs Meshy - Different Roles:**
> - **SAM3D** = Bulk scene initialization (reconstructs everything from image)
> - **Meshy** = On-demand quality upgrades (generates specific objects during refinement)
>
> They can be used together: SAM3D sets up the scene, Meshy improves individual assets.

## Pipeline Workflow (dynamic_scene mode)

### Stage 1: Input

User provides target image + text description of dynamic effects.

Components: Target Image, Description

### Stage 2: SAM3D Scene Initialization (Recommended)

**Enable with:** `--prompt-setting init --generator-tools tools/sam3d/init.py,...`

Components: SAM Segmentation (Auto-detect all objects), SAM3D to GLB (Image to 3D mesh), Import to Blender (Initial scene ready)

> **Without this step:** Agent starts from empty scene and uses only primitives (cubes, cylinders). SAM3D gives a much better starting point by reconstructing actual objects from the image.
>
> **Requirements:** Linux + 32GB VRAM GPU, or use **Meshy API** as alternative.

### Stage 3: GPT-4o (ChatGPT Vision) Analysis

**OpenAI API call:** Sends image + prompt to GPT-4o --> Returns Blender Python code

Components: Image Analysis (GPT-4o vision), Scene Planning (initialize_plan tool), Code Generation (execute_and_evaluate)

### Stage 3.5: (Optional) Meshy API - 3D Asset Generation

**Meshy API call:** When GPT-4o calls `get_better_object` tool --> Meshy generates detailed 3D model

Components: Text-to-3D ("green tea bottle"), GLB Export (With PBR textures), Import to Blender (Replace primitives)

### Stage 4: Dual-Agent Iterative Refinement

Generator creates code, Verifier evaluates renders, loop until satisfactory.

## Dual-Agent Architecture

### Generator Agent
Creates Blender Python code
- `initialize_plan`
- `execute_and_evaluate`
- `get_scene_info`
- `undo_last_step`
- `end`

### Verifier Agent
Evaluates render quality
- `investigate`
- `set_camera`
- `set_visibility`
- `set_keyframe`
- `reload_scene`

### Stage 5: Blender Execution

Python scripts executed in Blender to build geometry, materials, lighting, animation.

Components: Geometry, Materials, Lighting, Camera, Animation

### Stage 6: Output

Final 3D scene with animation ready for rendering.

Components: .blend File, Renders (PNG), Scripts

## Project Structure

```
VIGA/
├── agents/           # Generator & Verifier agent implementations
├── data/             # Input tasks, targets, scripts
│   ├── dynamic_scene/
│   ├── static_scene/
│   └── blendergym/
├── output/           # Generated results (.blend, renders, scripts)
├── prompts/          # System prompts for each mode
├── runners/          # Entry points (dynamic_scene.py, static_scene.py)
├── tools/            # MCP tool servers
│   ├── blender/      # Blender execution tools
│   └── sam3d/        # SAM segmentation tools
├── utils/
│   ├── third_party/
│   │   ├── infinigen/ # Procedural 3D generation
│   │   ├── sam/       # Segment Anything Model
│   │   ├── sam3/      # SAM version 3
│   │   └── sam3d/     # SAM 3D reconstruction
│   └── _api_keys.py  # API key configuration
└── main.py           # Main entry point
```

## Run Commands

```bash
# Dynamic Scene (VLM-based, what we tested)
python runners/dynamic_scene.py --task greentea --model gpt-4o \
    --blender-command "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \
    --max-rounds 20

# With SAM initialization (auto-detect objects)
python runners/dynamic_scene.py --task greentea --model gpt-4o \
    --prompt-setting init \
    --generator-tools "tools/sam3d/init.py,tools/blender/exec.py,tools/generator_base.py"
```

---

*Generated by Yuna | VIGA Workflow Analysis*
*January 29, 2026*
