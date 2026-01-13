#!/bin/bash
# 检查实验一进度的脚本
cd /root/autodl-tmp/ijcai
echo "=== IJCAI实验一运行状态 ==="
echo ""
echo "进程状态:"
ps aux | grep "[p]ython.*run_ijcai_experiments" | head -1 | awk '{print "  PID: "$2", CPU: "$3"%, 内存: "$6/1024"MB, 运行时间: "$10}'
echo ""
echo "最新进度（最后20行）:"
tail -20 results/ijcai_experiments/exp1.log 2>/dev/null | grep -E "方法:|预算:|Random|High-Degree|PageRank|FINDER|Semantic|seed|完成|错误" | tail -10 || tail -5 results/ijcai_experiments/exp1.log 2>/dev/null
echo ""
echo "输出文件:"
ls -lh results/ijcai_experiments/*.json 2>/dev/null | tail -3 || echo "  结果文件尚未生成"
echo ""
echo "GPU使用:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | awk '{print "  GPU利用率: "$1"%, 内存: "$2}'
