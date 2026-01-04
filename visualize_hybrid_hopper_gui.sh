#!/bin/bash

#################################################################
# Visualize Hybrid Hopper URDF
# GUI visualization to verify robot structure
#################################################################

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       🚁 Hybrid Hopper URDF Visualization (GUI)              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Set environment variables for GUI
export LD_LIBRARY_PATH=/home/abc/miniconda3/envs/walk_these_ways/lib:$LD_LIBRARY_PATH
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY=:0

# Enable X11 permissions
xhost + > /dev/null 2>&1

echo ""
echo "Launching GUI visualization..."
echo "Press Ctrl+C to stop"
echo ""

# Change to project directory
cd /home/abc/walk_these_ways_learning || exit 1

# Run visualization
/home/abc/miniconda3/envs/walk_these_ways/bin/python visualize_hybrid_hopper.py

echo ""
echo "Visualization closed."

