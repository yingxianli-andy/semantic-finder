# Semantic-FINDER: 语义增强的关键节点发现与干预

基于强化学习和 LLM 的社交网络去极化系统。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python -m experiments.train --num_episodes 1000 --device cpu
```

### 测试模型
```bash
python -m experiments.test --model_path results/models/final_model.pth --dataset_name ego-Twitter
```

## 项目结构

- `src/agent/`: RL Agent 代码（安迪）
- `src/environment/`: 环境模拟器（哈工爷）
- `src/llm/`: LLM 接口（上交爷）
- `experiments/`: 训练和测试脚本
- `config/`: 配置文件

详细文档请参考 `doc/` 目录。

