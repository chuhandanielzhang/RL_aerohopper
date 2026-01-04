#!/bin/bash

#################################################################
# Hybrid Hopper Training Script
# Single leg + quadrotor hybrid robot training
#################################################################

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          🚁 Starting Hybrid Hopper Training                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Set environment variables
export PATH=/home/abc/miniconda3/envs/walk_these_ways/bin:$PATH
export LD_LIBRARY_PATH=/home/abc/miniconda3/envs/walk_these_ways/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0  # Enable async for speed (set to 1 if debugging CUDA errors)
export MAX_JOBS=4  # Limit parallel compilation

# Print system info
echo ""
echo "System Information:"
echo "─────────────────────────────────────────────────────────────"
echo "CUDA Device: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0)"
echo "Python: $(which python)"
echo "─────────────────────────────────────────────────────────────"
echo ""

# Change to project directory
cd /home/abc/walk_these_ways_learning/walk-these-ways || exit 1

# Activate conda environment
source /home/abc/miniconda3/etc/profile.d/conda.sh
conda activate walk_these_ways

echo ""
echo "Training Configuration:"
echo "─────────────────────────────────────────────────────────────"
echo "• Robot: Hybrid Hopper (1 leg + 4 rotors)"
echo "• DOF: 7 (2 ball joints + 1 spring + 4 rotors)"
echo "• Rotor Curriculum: Progressive reduction"
echo "  - Phase 1 (0-500):    100% rotor assist"
echo "  - Phase 2 (500-1500):  70% rotor assist"
echo "  - Phase 3 (1500-3000): 40% rotor assist"
echo "  - Phase 4 (3000+):     20% rotor assist"
echo "• Max Iterations: 5000"
echo "• Environments: 1024"
echo "─────────────────────────────────────────────────────────────"
echo ""

# Kill any existing training processes
pkill -f "python.*train.py.*hybrid_hopper" 2>/dev/null
sleep 1

echo "🚀 Starting training..."
echo ""

# Run training (headless mode)
python scripts/train.py hybrid_hopper

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          Training Complete or Interrupted                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

