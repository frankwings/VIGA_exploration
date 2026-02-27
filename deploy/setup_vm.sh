#!/bin/bash
# VIGA/SAM3D GCP VM Setup Script
# Target: g2-standard-8 (L4 GPU, 24GB VRAM), Debian 12, CUDA 12.x
# Usage: bash setup_vm.sh [stage]
#   stage 1: conda envs (sam3d_py311, agent, sam, blender)
#   stage 2: model weights + Blender
#   stage 3: _api_keys.py + smoke test
#   (no arg = run all stages)

set -euo pipefail

VIGA_ROOT="$HOME/GenesisVIGA"
CONDA="$HOME/miniconda3/bin/conda"
PIP="pip"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ─── Stage 1: Conda Environments ──────────────────────────────────────────────

stage1() {
  log "=== Stage 1: Creating conda environments ==="

  # ── sam3d_py311 (hardest — pytorch3d, kaolin, spconv, flash_attn) ──
  if ! $CONDA env list | grep -q sam3d_py311; then
    log "Creating sam3d_py311 env..."
    $CONDA create -n sam3d_py311 python=3.11 -y -q
  fi

  log "Installing sam3d_py311 packages..."
  eval "$($CONDA shell.bash hook)"
  conda activate sam3d_py311

  # PyTorch 2.5.1 + CUDA 12.1 (matches spconv-cu121 from requirements.txt)
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

  # Core deps from sam3d requirements
  pip install \
    numpy scipy pillow opencv-python trimesh open3d \
    hydra-core omegaconf loguru einops timm transformers diffusers \
    safetensors huggingface-hub accelerate \
    scikit-image scikit-learn matplotlib seaborn \
    plyfile pygltflib roma fire pyyaml \
    kornia imageio tqdm requests ninja pybind11 \
    2>&1 | tail -5

  # spconv (CUDA 12.1)
  pip install spconv-cu121==2.3.8 2>&1 | tail -3

  # utils3d
  pip install utils3d 2>&1 | tail -3

  # MoGe (from git)
  pip install "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" 2>&1 | tail -3

  # gsplat (from git, matches requirements.inference.txt)
  # --no-build-isolation: setup.py imports torch
  pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" 2>&1 | tail -5

  # pytorch3d (build from source — this is the slow one, ~10-20 min)
  # --no-build-isolation: setup.py imports torch
  log "Building pytorch3d from source (this takes ~15 min)..."
  pip install --no-build-isolation "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" 2>&1 | tail -5

  # flash_attn (build from source)
  log "Building flash_attn from source (this takes ~10 min)..."
  pip install flash_attn==2.8.3 --no-build-isolation 2>&1 | tail -5

  # kaolin
  log "Installing kaolin..."
  pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html 2>&1 | tail -3

  # Install sam3d_objects package (the SAM3D pipeline itself)
  cd "$VIGA_ROOT/utils/third_party/sam3d"
  pip install -e . 2>&1 | tail -3

  conda deactivate
  log "sam3d_py311 done."

  # ── agent (Python 3.10 — main VIGA pipeline) ──
  if ! $CONDA env list | grep -q "^agent "; then
    log "Creating agent env..."
    $CONDA create -n agent python=3.10 -y -q
  fi

  log "Installing agent packages..."
  conda activate agent

  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
  pip install \
    openai pillow requests numpy scipy tqdm pyyaml \
    trimesh thefuzz python-Levenshtein \
    2>&1 | tail -3

  # Install VIGA models requirements
  pip install -r "$VIGA_ROOT/models/requirements.txt" 2>&1 | tail -3

  conda deactivate
  log "agent done."

  # ── sam (Python 3.10 — SAM segmentation) ──
  if ! $CONDA env list | grep -q "^sam "; then
    log "Creating sam env..."
    $CONDA create -n sam python=3.10 -y -q
  fi

  log "Installing sam packages..."
  conda activate sam

  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
  pip install numpy pillow opencv-python matplotlib requests openai 2>&1 | tail -3

  # Install segment-anything
  cd "$VIGA_ROOT/utils/third_party/sam"
  pip install -e . 2>&1 | tail -3

  conda deactivate
  log "sam done."

  # ── blender (Python 3.11 — Blender scripts) ──
  if ! $CONDA env list | grep -q "^blender "; then
    log "Creating blender env..."
    $CONDA create -n blender python=3.11 -y -q
  fi

  log "Installing blender packages..."
  conda activate blender
  pip install numpy pillow trimesh 2>&1 | tail -3
  conda deactivate
  log "blender done."

  log "=== Stage 1 complete ==="
}

# ─── Stage 2: Model Weights + Blender ─────────────────────────────────────────

stage2() {
  log "=== Stage 2: Model weights + Blender ==="

  # SAM ViT-H checkpoint
  SAM_CKPT="$VIGA_ROOT/utils/third_party/sam/sam_vit_h_4b8939.pth"
  if [ ! -f "$SAM_CKPT" ]; then
    log "Downloading SAM ViT-H (2.5 GB)..."
    curl -fSL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o "$SAM_CKPT"
  else
    log "SAM ViT-H already exists."
  fi

  # TRELLIS + MoGe + DINOv2 are auto-downloaded on first run via HuggingFace Hub.
  # Pre-download TRELLIS to avoid first-run delay:
  log "Pre-downloading TRELLIS 2.0 model (~3 GB)..."
  eval "$($CONDA shell.bash hook)"
  conda activate sam3d_py311
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/TRELLIS.2-4B', local_dir=None)
print('TRELLIS downloaded to HF cache')
" 2>&1 | tail -5
  conda deactivate

  # Blender 4.5 LTS headless
  if ! which blender >/dev/null 2>&1 && [ ! -f /opt/blender/blender ]; then
    log "Installing Blender 4.5..."
    BLENDER_VER="4.5.0"
    BLENDER_TAR="blender-${BLENDER_VER}-linux-x64.tar.xz"
    BLENDER_URL="https://mirror.clarkson.edu/blender/release/Blender4.5/${BLENDER_TAR}"
    cd /tmp
    curl -fSL "$BLENDER_URL" -o "$BLENDER_TAR"
    sudo mkdir -p /opt/blender
    sudo tar xf "$BLENDER_TAR" -C /opt/blender --strip-components=1
    sudo ln -sf /opt/blender/blender /usr/local/bin/blender
    rm -f "$BLENDER_TAR"
    log "Blender installed: $(blender --version 2>&1 | head -1)"
  else
    log "Blender already installed."
  fi

  # Xvfb for headless EEVEE rendering (needs OpenGL context)
  if ! which Xvfb >/dev/null 2>&1; then
    log "Installing Xvfb for headless rendering..."
    sudo apt-get update -qq && sudo apt-get install -y -qq xvfb libgl1-mesa-glx libglu1-mesa 2>&1 | tail -3
  fi

  log "=== Stage 2 complete ==="
}

# ─── Stage 3: API Keys + Paths + Smoke Test ───────────────────────────────────

stage3() {
  log "=== Stage 3: API keys + path config + smoke test ==="

  # Create utils/_path.py for Linux conda paths
  CONDA_BASE="$HOME/miniconda3/envs"
  cat > "$VIGA_ROOT/utils/_path_linux.py" << PYEOF
"""Linux path overrides for GCP VM deployment."""
import os

CONDA_BASE = os.path.expanduser("~/miniconda3/envs")

PYTHON_PATHS = {
    "agent":       os.path.join(CONDA_BASE, "agent", "bin", "python"),
    "blender":     os.path.join(CONDA_BASE, "blender", "bin", "python"),
    "sam":         os.path.join(CONDA_BASE, "sam", "bin", "python"),
    "sam3d_py311": os.path.join(CONDA_BASE, "sam3d_py311", "bin", "python"),
}

BLENDER_CMD = "/usr/local/bin/blender"
PYEOF

  # Check if _api_keys.py exists
  if [ ! -f "$VIGA_ROOT/utils/_api_keys.py" ]; then
    log "WARNING: utils/_api_keys.py missing! Copy from Windows machine."
  else
    log "API keys file exists."
  fi

  # Smoke test: check all conda envs can import torch
  eval "$($CONDA shell.bash hook)"
  for env in agent sam sam3d_py311; do
    conda activate "$env"
    RESULT=$(python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>&1)
    log "  $env: $RESULT"
    conda deactivate
  done

  # Test nvidia-smi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

  log "=== Stage 3 complete ==="
  log ""
  log "Next steps:"
  log "  1. Copy utils/_api_keys.py from Windows"
  log "  2. Update utils/_path.py for Linux paths"
  log "  3. Run a test: python runners/static_scene.py --task=greentea --model=gpt-5 ..."
}

# ─── Main ─────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"
case "$STAGE" in
  1) stage1 ;;
  2) stage2 ;;
  3) stage3 ;;
  all) stage1; stage2; stage3 ;;
  *) echo "Usage: $0 [1|2|3|all]"; exit 1 ;;
esac
