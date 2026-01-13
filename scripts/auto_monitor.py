#!/usr/bin/env python3
"""
自动监控训练脚本

持续监控训练过程，自动检测：
- 训练是否在运行
- 训练进度
- 错误和异常
- 资源使用情况
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.models_dir = project_root / "results" / "models"
        self.log_file = project_root / "training_log.txt"
        self.last_checkpoint_time = None
        self.last_log_size = 0
        self.stall_count = 0
        
    def check_training_process(self):
        """检查训练进程是否在运行"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "train.py"],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip()) > 0
        except:
            return False
    
    def get_latest_checkpoint(self):
        """获取最新检查点"""
        if not self.models_dir.exists():
            return None
        
        checkpoints = list(self.models_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=os.path.getmtime)
        return {
            "name": latest.name,
            "path": str(latest),
            "time": datetime.fromtimestamp(os.path.getmtime(latest)),
            "size": os.path.getsize(latest) / 1024 / 1024  # MB
        }
    
    def get_training_stats(self):
        """获取训练统计信息"""
        stats = {
            "process_running": self.check_training_process(),
            "latest_checkpoint": self.get_latest_checkpoint(),
            "log_exists": self.log_file.exists(),
            "log_size": 0,
            "log_growing": False
        }
        
        if self.log_file.exists():
            stats["log_size"] = os.path.getsize(self.log_file) / 1024  # KB
            stats["log_growing"] = stats["log_size"] > self.last_log_size
            self.last_log_size = stats["log_size"]
        
        return stats
    
    def print_status(self, stats):
        """打印状态"""
        print("\n" + "=" * 70)
        print(f"训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # 进程状态
        if stats["process_running"]:
            print("✓ 训练进程: 运行中")
        else:
            print("✗ 训练进程: 未运行")
        
        # 检查点信息
        if stats["latest_checkpoint"]:
            cp = stats["latest_checkpoint"]
            print(f"\n✓ 最新检查点: {cp['name']}")
            print(f"  时间: {cp['time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  大小: {cp['size']:.2f} MB")
            
            # 检查是否卡住
            if self.last_checkpoint_time:
                time_diff = (cp['time'] - self.last_checkpoint_time).total_seconds()
                if time_diff > 300:  # 5分钟没有新检查点
                    self.stall_count += 1
                    if self.stall_count > 3:
                        print(f"  ⚠ 警告: 超过 {time_diff/60:.1f} 分钟没有新检查点")
                else:
                    self.stall_count = 0
            self.last_checkpoint_time = cp['time']
        else:
            print("\n⚠ 未找到检查点")
        
        # 日志信息
        if stats["log_exists"]:
            status = "增长中" if stats["log_growing"] else "未增长"
            print(f"\n✓ 训练日志: {status} ({stats['log_size']:.2f} KB)")
        else:
            print("\n⚠ 未找到训练日志")
        
        # 读取最后几行日志
        if stats["log_exists"]:
            try:
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("\n最后几行日志:")
                        for line in lines[-3:]:
                            print(f"  {line.rstrip()}")
            except:
                pass
        
        print("=" * 70)
    
    def monitor(self, interval=10, max_iterations=None):
        """监控循环"""
        print("开始自动监控训练过程...")
        print("按 Ctrl+C 停止监控\n")
        
        iteration = 0
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    break
                
                stats = self.get_training_stats()
                self.print_status(stats)
                
                # 检查是否卡住
                if not stats["process_running"] and stats["latest_checkpoint"]:
                    print("\n⚠ 训练进程已停止，但检查点存在")
                
                iteration += 1
                if max_iterations is None or iteration < max_iterations:
                    print(f"\n等待 {interval} 秒后刷新...")
                    time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n监控已停止")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自动监控训练")
    parser.add_argument("--interval", type=int, default=10, help="刷新间隔（秒）")
    parser.add_argument("--iterations", type=int, default=None, help="最大迭代次数")
    parser.add_argument("--once", action="store_true", help="只检查一次")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor()
    
    if args.once:
        stats = monitor.get_training_stats()
        monitor.print_status(stats)
    else:
        monitor.monitor(args.interval, args.iterations)

