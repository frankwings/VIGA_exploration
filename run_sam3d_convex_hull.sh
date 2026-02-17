#!/bin/bash
set -e

INPUT_DIR="output/test/greentea/sam_init"
OUTPUT_DIR="output/sam3d_convex_hull_v2"
CONFIG="utils/third_party/sam3d/checkpoints/hf/checkpoints/pipeline.yaml"
SCENE_IMAGE="data/static_scene/greentea/target.png"

OBJECTS=("green_tea_bottle" "ito_en_bottle" "alienware_keyboard" "headphones" "envelope")

echo "========================================"
echo "SAM3D Batch Run - Convex Hull + Denormalized Pointmap"
echo "========================================"
echo "Start time: $(date)"
echo ""

for obj in "${OBJECTS[@]}"; do
    echo "============================================================"
    echo "Processing: $obj"
    echo "Start: $(date)"
    echo "============================================================"

    conda run -n sam3d_py311 python tools/sam3d/sam3d_worker.py \
        --image "$INPUT_DIR/$obj.png" \
        --mask "$INPUT_DIR/$obj.npy" \
        --config "$CONFIG" \
        --glb "$OUTPUT_DIR/$obj.glb" \
        --info "$OUTPUT_DIR/${obj}_info.json" \
        --scene-image "$SCENE_IMAGE" \
        2>&1 | tee "$OUTPUT_DIR/${obj}.log"

    echo "Finished $obj at $(date)"
    echo ""
done

echo "========================================"
echo "ALL DONE at $(date)"
echo "========================================"
