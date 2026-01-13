#!/usr/bin/env python3
"""
数据集下载脚本

支持多种下载方式：
1. 直接下载（如果网络好）
2. 使用镜像站（国内镜像）
3. 使用代理（如果配置了）
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# SNAP 数据集 URL（主要和镜像）
DATASETS = {
    "ego-Twitter": {
        "primary": "https://snap.stanford.edu/data/twitter.tar.gz",
        "mirror": None,  # 可以添加镜像站 URL
        "filename": "twitter.tar.gz"
    },
    # 可以添加更多数据集
}

DATA_DIR = project_root / "data" / "raw"


def download_with_wget(url: str, output_path: Path, use_proxy: bool = False):
    """使用 wget 下载"""
    cmd = ["wget", "-q", "--show-progress", url, "-O", str(output_path)]
    
    if use_proxy:
        # 如果配置了代理，可以在这里设置
        # 例如：cmd = ["wget", "-e", "http_proxy=http://proxy:port", ...]
        pass
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"下载超时: {url}")
        return False
    except Exception as e:
        print(f"wget 下载失败: {e}")
        return False


def download_with_urllib(url: str, output_path: Path, use_proxy: bool = False):
    """使用 urllib 下载"""
    try:
        if use_proxy:
            # 配置代理
            proxy_handler = urllib.request.ProxyHandler({
                'http': 'http://proxy:port',
                'https': 'https://proxy:port'
            })
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, str(output_path))
        return True
    except Exception as e:
        print(f"urllib 下载失败: {e}")
        return False


def download_dataset(dataset_name: str, use_mirror: bool = False, use_proxy: bool = False):
    """下载数据集"""
    if dataset_name not in DATASETS:
        print(f"错误: 未知的数据集名称: {dataset_name}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return False
    
    dataset_info = DATASETS[dataset_name]
    
    # 选择 URL
    if use_mirror and dataset_info.get("mirror"):
        url = dataset_info["mirror"]
        print(f"使用镜像站下载: {url}")
    else:
        url = dataset_info["primary"]
        print(f"使用主站下载: {url}")
    
    # 确保输出目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / dataset_info["filename"]
    
    # 检查文件是否已存在
    if output_path.exists():
        print(f"文件已存在: {output_path}")
        response = input("是否重新下载？(y/n): ").strip().lower()
        if response != 'y':
            print("跳过下载")
            return True
    
    print(f"开始下载 {dataset_name}...")
    print(f"保存到: {output_path}")
    
    # 尝试使用 wget
    if download_with_wget(url, output_path, use_proxy):
        print(f"✓ 下载成功: {output_path}")
        return True
    
    # 如果 wget 失败，尝试 urllib
    print("wget 失败，尝试使用 urllib...")
    if download_with_urllib(url, output_path, use_proxy):
        print(f"✓ 下载成功: {output_path}")
        return True
    
    print(f"✗ 下载失败: {dataset_name}")
    print("\n建议:")
    print("1. 检查网络连接")
    print("2. 尝试使用镜像站（如果有）")
    print("3. 配置代理（如果服务器支持）")
    print("4. 手动下载后放到 data/raw/ 目录")
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 SNAP 数据集")
    parser.add_argument("--dataset", type=str, default="ego-Twitter",
                       choices=list(DATASETS.keys()),
                       help="要下载的数据集名称")
    parser.add_argument("--use-mirror", action="store_true",
                       help="使用镜像站下载")
    parser.add_argument("--use-proxy", action="store_true",
                       help="使用代理下载")
    
    args = parser.parse_args()
    
    success = download_dataset(args.dataset, args.use_mirror, args.use_proxy)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

