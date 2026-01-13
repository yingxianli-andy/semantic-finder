# 数据集下载指南

## 方法 1: 使用下载脚本（推荐）

```bash
# 基本用法
python scripts/download_datasets.py --dataset ego-Twitter

# 如果网络不好，可以尝试多次
python scripts/download_datasets.py --dataset ego-Twitter
```

## 方法 2: 手动下载

### 2.1 直接下载（如果网络好）

```bash
cd /root/autodl-tmp/ijcai/data/raw
wget https://snap.stanford.edu/data/ego-Twitter.txt.gz
```

### 2.2 使用镜像站（国内服务器推荐）

如果 Stanford 的服务器访问慢，可以尝试：

1. **清华大学镜像**（如果有）：
```bash
wget https://mirrors.tuna.tsinghua.edu.cn/snap/ego-Twitter.txt.gz
```

2. **其他镜像站**：可以在网上搜索 "SNAP dataset mirror" 或 "Stanford SNAP 镜像"

### 2.3 配置代理（autodl 翻墙方法）

根据你 boss 的建议，autodl 服务器可能需要配置代理。常见方法：

#### 方法 A: 使用环境变量

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
wget https://snap.stanford.edu/data/ego-Twitter.txt.gz
```

#### 方法 B: 使用 proxychains

```bash
# 安装 proxychains
apt-get install proxychains4

# 配置代理（编辑 /etc/proxychains4.conf）
# 然后使用：
proxychains4 wget https://snap.stanford.edu/data/ego-Twitter.txt.gz
```

#### 方法 C: 使用 autodl 的内置代理（如果有）

有些 autodl 服务器已经配置了代理，可以直接使用。

## 方法 3: 本地下载后上传

如果服务器网络实在不行：

1. 在你的本地电脑上下载数据集：
   - 访问：https://snap.stanford.edu/data/
   - 下载 `ego-Twitter.txt.gz`

2. 上传到服务器：
```bash
# 使用 scp（在你的本地电脑上运行）
scp ego-Twitter.txt.gz user@your-server:/root/autodl-tmp/ijcai/data/raw/
```

## 验证下载

下载完成后，检查文件：

```bash
ls -lh /root/autodl-tmp/ijcai/data/raw/
```

应该能看到 `ego-Twitter.txt.gz` 文件。

## 解压（如果需要）

```bash
cd /root/autodl-tmp/ijcai/data/raw
gunzip ego-Twitter.txt.gz
```

## 常见问题

1. **下载速度慢**：尝试使用镜像站或配置代理
2. **连接超时**：检查网络连接，或使用本地下载后上传的方法
3. **文件损坏**：重新下载

## 参考链接

- SNAP 数据集主页：https://snap.stanford.edu/data/
- ego-Twitter 数据集：https://snap.stanford.edu/data/ego-Twitter.html
