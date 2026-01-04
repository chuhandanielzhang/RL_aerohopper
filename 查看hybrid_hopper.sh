#!/bin/bash
# Hybrid Hopper - URDF可视化启动脚本

echo "🚀 启动Hybrid Hopper可视化..."
echo ""

# 激活conda环境
source /home/abc/miniconda3/etc/profile.d/conda.sh
conda activate walk_these_ways

# 设置Isaac Gym环境变量
export LD_LIBRARY_PATH=/home/abc/miniconda3/envs/walk_these_ways/lib:$LD_LIBRARY_PATH

# 切换到项目目录
cd /home/abc/walk_these_ways_learning

# 运行可视化
echo "📋 URDF文件: walk-these-ways/resources/robots/hybrid_hopper/urdf/hybrid_hopper.urdf"
echo ""
echo "⏳ 正在加载..."
echo ""

python visualize_hybrid_hopper_simple.py

echo ""
echo "✅ 可视化结束"

