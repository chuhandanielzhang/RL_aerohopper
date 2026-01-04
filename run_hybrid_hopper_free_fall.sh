#!/bin/bash

#################################################################
# Hybrid Hopper Free Fall Test Script
# Launch GUI test with proper environment setup
#################################################################

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       🚁 Hybrid Hopper Free Fall Test (GUI)                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Set environment variables for GUI
export LD_LIBRARY_PATH=/home/abc/miniconda3/envs/walk_these_ways/lib:$LD_LIBRARY_PATH
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY=:0

# Enable X11 permissions
xhost + > /dev/null 2>&1

echo ""
echo "测试配置:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• 3个机器人，不同初始姿态"
echo "• 从3米高度自由落体"
echo "• 无主动控制（纯物理）"
echo "• 观察：旋翼、关节、弹簧行为"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Kill any previous visualization
pkill -f "python.*visualize_hybrid" 2>/dev/null
sleep 1

# Change to project directory
cd /home/abc/walk_these_ways_learning || exit 1

# Run free fall test
/home/abc/miniconda3/envs/walk_these_ways/bin/python test_hybrid_hopper_free_fall.py

echo ""
echo "Test complete."

