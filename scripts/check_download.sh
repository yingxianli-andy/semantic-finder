#!/bin/bash
# 检查下载进度脚本

FILE="/root/autodl-tmp/ijcai/data/raw/ego-Twitter.tar.gz"
PID=$(pgrep -f "wget.*twitter.tar.gz" | head -1)

echo "=== 下载进度检查 ==="
echo ""

# 检查进程是否在运行
if [ -n "$PID" ]; then
    echo "✓ 下载进程正在运行 (PID: $PID)"
else
    echo "✗ 下载进程未运行（可能已完成或失败）"
fi

echo ""

# 检查文件是否存在
if [ -f "$FILE" ]; then
    SIZE=$(ls -lh "$FILE" | awk '{print $5}')
    SIZE_BYTES=$(stat -c%s "$FILE" 2>/dev/null || du -b "$FILE" 2>/dev/null | awk '{print $1}')
    
    echo "文件: $FILE"
    echo "当前大小: $SIZE ($SIZE_BYTES 字节)"
    echo ""
    
    # 估算总大小（twitter.tar.gz 通常约 20-50MB，但不确定）
    echo "提示: twitter.tar.gz 通常大小在 20-50MB 左右"
    echo ""
    
    # 检查文件是否还在增长
    sleep 2
    NEW_SIZE_BYTES=$(stat -c%s "$FILE" 2>/dev/null || du -b "$FILE" 2>/dev/null | awk '{print $1}')
    
    if [ "$SIZE_BYTES" -lt "$NEW_SIZE_BYTES" ]; then
        DIFF=$((NEW_SIZE_BYTES - SIZE_BYTES))
        echo "✓ 文件正在增长中（2秒内增加了 $DIFF 字节）"
    elif [ "$SIZE_BYTES" -eq "$NEW_SIZE_BYTES" ]; then
        if [ -n "$PID" ]; then
            echo "⚠ 文件大小未变化（可能下载很慢或卡住）"
        else
            echo "✓ 下载可能已完成"
        fi
    fi
else
    echo "✗ 文件不存在"
fi

echo ""
echo "=== 网络连接检查 ==="
# 检查到 Stanford 的连接
if timeout 3 curl -s -I https://snap.stanford.edu/data/twitter.tar.gz | head -1 | grep -q "200\|206"; then
    echo "✓ 可以连接到下载服务器"
else
    echo "✗ 无法连接到下载服务器（可能网络问题）"
fi
