#!/usr/bin/env python3
"""
训练监控脚本

实时监控训练过程，包括：
- 训练进度
- 损失和奖励曲线
- GPU/CPU 使用情况
- 模型检查点
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_training_status():
    """检查训练状态"""
    models_dir = project_root / "results" / "models"
    logs_dir = project_root / "results" / "logs"
    
    status = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "latest_checkpoint": None,
        "training_active": False
    }
    
    # 检查模型文件
    if models_dir.exists():
        model_files = sorted(models_dir.glob("*.pth"), key=os.path.getmtime, reverse=True)
        status["models"] = [f.name for f in model_files[:5]]  # 最近5个
        if model_files:
            status["latest_checkpoint"] = model_files[0].name
            status["latest_checkpoint_time"] = datetime.fromtimestamp(
                os.path.getmtime(model_files[0])
            ).strftime("%Y-%m-%d %H:%M:%S")
            status["latest_checkpoint_size"] = f"{os.path.getsize(model_files[0]) / 1024 / 1024:.2f} MB"
    
    # 检查训练日志
    log_file = project_root / "training_log.txt"
    if log_file.exists():
        status["training_active"] = True
        status["log_file"] = str(log_file)
        status["log_size"] = f"{os.path.getsize(log_file) / 1024:.2f} KB"
        
        # 读取最后几行
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                status["last_lines"] = lines[-5:] if len(lines) > 5 else lines
        except:
            pass
    
    return status


def print_status(status):
    """打印状态信息"""
    print("=" * 60)
    print(f"训练监控 - {status['timestamp']}")
    print("=" * 60)
    
    if status["latest_checkpoint"]:
        print(f"\n✓ 最新检查点: {status['latest_checkpoint']}")
        print(f"  时间: {status['latest_checkpoint_time']}")
        print(f"  大小: {status['latest_checkpoint_size']}")
    else:
        print("\n⚠ 未找到检查点")
    
    if status["models"]:
        print(f"\n最近模型文件 ({len(status['models'])} 个):")
        for i, model in enumerate(status["models"][:3], 1):
            print(f"  {i}. {model}")
    
    if status["training_active"]:
        print(f"\n✓ 训练日志存在: {status['log_file']}")
        print(f"  大小: {status['log_size']}")
        if status.get("last_lines"):
            print("\n最后几行日志:")
            for line in status["last_lines"]:
                print(f"  {line.rstrip()}")
    else:
        print("\n⚠ 未检测到训练日志")
    
    print("=" * 60)


def monitor_loop(interval=10):
    """监控循环"""
    print("开始监控训练过程...")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            status = check_training_status()
            print_status(status)
            print(f"\n等待 {interval} 秒后刷新...\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练监控脚本")
    parser.add_argument("--interval", type=int, default=10, help="刷新间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只检查一次，不循环")
    
    args = parser.parse_args()
    
    if args.once:
        status = check_training_status()
        print_status(status)
    else:
        monitor_loop(args.interval)

