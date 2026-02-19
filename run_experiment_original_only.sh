#!/bin/bash
# Run just the original SAM3D experiment (after VIGA run completes)
# Execute from project root

set -e

PYTHON="C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe"
PROJECT_ROOT="/d/Projects/ProjectGenesis/GenesisVIGA"
SAM3D_SUB="$PROJECT_ROOT/utils/third_party/sam3d"
ORIGINAL_COMMIT="af582ce"
VIGA_COMMIT="5b667c8"

OUT_DIR="$PROJECT_ROOT/output/experiment_original_sam3d"
CONFIG="$SAM3D_SUB/checkpoints/hf/checkpoints/pipeline.yaml"
# Use resized image (771x1024) — matches mask dims from SAM segmentation
TARGET_IMAGE="$PROJECT_ROOT/data/static_scene/dining/target_resized.jpg"
MASK_FILE="$PROJECT_ROOT/output/sam3d_dining/wooden_chair.npy"
GLB_PATH="$OUT_DIR/wooden_chair.glb"
INFO_PATH="$OUT_DIR/wooden_chair_info.json"
LOG_PATH="$OUT_DIR/wooden_chair_log.txt"

mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Checking out SAM3D submodule to $ORIGINAL_COMMIT..."
git -C "$SAM3D_SUB" checkout "$ORIGINAL_COMMIT"

echo "[$(date '+%H:%M:%S')] Running original SAM3D worker..."
"$PYTHON" -u "$PROJECT_ROOT/_exp_worker_original.py" \
  --image "$TARGET_IMAGE" \
  --mask "$MASK_FILE" \
  --config "$CONFIG" \
  --glb "$GLB_PATH" \
  --info "$INFO_PATH" \
  > "$LOG_PATH" 2>&1
STATUS=$?

echo "[$(date '+%H:%M:%S')] Restoring submodule to $VIGA_COMMIT..."
git -C "$SAM3D_SUB" checkout "$VIGA_COMMIT"

if [ $STATUS -eq 0 ]; then
  echo "[$(date '+%H:%M:%S')] Original run SUCCEEDED -> $OUT_DIR"
else
  echo "[$(date '+%H:%M:%S')] Original run FAILED (exit $STATUS) — see $LOG_PATH"
  tail -20 "$LOG_PATH"
fi
