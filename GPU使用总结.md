# GPU使用总结

**时间**: 2024-01-09  
**问题**: 为什么之前测试没有使用GPU？

---

## 🔍 问题原因

### 为什么之前没用GPU？

1. **测试命令没有指定 `--device cuda`**
   - 测试脚本默认使用 `cpu`
   - 需要显式指定 `--device cuda` 才能使用GPU

2. **哈工爷已重构支持GPU**
   - ✅ `dynamics.py` 已支持GPU加速
   - ✅ 使用 `torch.sparse` 进行GPU计算
   - ✅ 有自动回退机制

3. **但调用时没有传递正确的device参数**
   - 之前的测试可能使用了默认的 `cpu`
   - 或者没有在命令行指定 `--device cuda`

---

## ✅ 当前状态

### GPU硬件

- ✅ **2个 RTX 4090 GPU**
- ✅ **每个 23.53 GB 显存**
- ✅ **CUDA 12.8 可用**

### GPU测试

- ✅ **GPU测试已启动**
- ✅ **使用 `--device cuda`**
- ✅ **进程正在运行**（已运行1分05秒）

---

## 📊 性能对比

### 小规模数据（1000节点）

- **CPU**: 0.0155 秒
- **GPU**: 0.3085 秒
- **结论**: CPU更快（GPU开销大）

### 大规模数据（10000节点）

- **CPU**: 待测试
- **GPU**: 待测试
- **预期**: GPU应该更快

### 超大规模数据（76K节点，twitter_combined.txt）

- **CPU**: 每个episode 10-30 分钟
- **GPU**: 每个episode 预计 1-5 分钟
- **预期加速**: 5-10倍

---

## 🚀 解决方案

### 1. 使用GPU运行测试

```bash
# 必须显式指定 --device cuda
python experiments/test.py \
    --model_path results/models/final_model.pth \
    --dataset_name twitter_combined.txt \
    --budget 5 \
    --n_runs 1 \
    --device cuda  # ← 关键：必须指定这个参数
```

### 2. 使用GPU运行训练

```bash
# 训练时也使用GPU
python experiments/train.py \
    --num_episodes 10000 \
    --device cuda  # ← 关键：必须指定这个参数
```

### 3. 根据数据规模选择

- **小规模** (<1000节点): 使用CPU（更快）
- **大规模** (>10000节点): 使用GPU（更快）

---

## 💡 关键点

### 为什么之前没用GPU？

**答案**: 因为测试命令没有指定 `--device cuda`！

- 测试脚本默认使用 `cpu`
- 即使有GPU，也需要显式指定 `--device cuda`
- 哈工爷的代码已经支持GPU，但调用时没有使用

### 如何确保使用GPU？

1. **命令行指定**:
   ```bash
   --device cuda
   ```

2. **检查GPU使用**:
   ```bash
   nvidia-smi  # 查看GPU使用情况
   ```

3. **代码中检查**:
   ```python
   if torch.cuda.is_available():
       device = "cuda"
   else:
       device = "cpu"
   ```

---

## ✅ 当前GPU测试状态

- ✅ **GPU测试进程运行中**（PID: 108900）
- ✅ **使用CUDA设备**
- ✅ **数据集**: twitter_combined.txt (76K节点)
- ⏳ **等待测试完成**

---

## 📈 预期效果

### 对于76K节点的大数据集

- **CPU**: 每个episode 10-30 分钟
- **GPU**: 每个episode 预计 1-5 分钟
- **加速比**: 预计 5-10倍

### 训练阶段

- **CPU**: ~2 it/s
- **GPU**: 预计 20-200 it/s
- **加速比**: 预计 10-100倍

---

## 🎯 总结

**问题**: 之前测试没有使用GPU  
**原因**: 测试命令没有指定 `--device cuda`  
**解决**: 现在已启动GPU测试，预计会有显著加速！

---

**最后更新**: 2024-01-09 18:27

