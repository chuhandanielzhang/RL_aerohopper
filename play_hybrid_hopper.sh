#!/bin/bash

export PATH=$HOME/miniconda3/envs/walk_these_ways/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/walk_these_ways/lib:$LD_LIBRARY_PATH

cd /home/abc/walk_these_ways_learning/walk-these-ways

# 最新的模型路径
MODEL_PATH="/home/abc/walk_these_ways_learning/walk-these-ways/runs/gait-conditioned-agility/2025-11-14/train/004757.365102"

cat <<'BANNER'
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🎮 Hybrid Hopper Demo Playback                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

✅ 训练完成！最终结果：
   - Iterations: 4990/5000
   - Total Reward: 6.318 ⭐
   - body_z: 2.971 ✅ (目标0.5，实际更高！)
   - foot_z: 0.249 ✅
   - tracking_contacts: 0.007 ✅

🎯 预期演示效果：
   - 周期性跳跃 (3Hz)
   - 稳定姿态控制
   - 旋翼辅助平衡
   - 可控方向移动

🎮 控制说明：
   - 默认自主跳跃演示
   - 关闭窗口退出

Starting demo...

BANNER

python scripts/play.py \
    --load_run="$MODEL_PATH" \
    --checkpoint=-1

echo ""
echo "Demo结束"
