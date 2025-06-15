# 中文仇恨识别模型训练

本项目使用Qwen-0.6B-Base模型，通过LoRA方法进行微调，实现中文仇恨识别任务。

## 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据：
- 将原始数据文件（CSV或JSON格式）放在项目根目录
- 数据格式应包含 `text` 和 `label` 两列
- 运行数据处理脚本：
```bash
python process_data.py
```

## 模型训练

1. 配置训练参数：
- 在 `train.py` 中可以调整以下参数：
  - `MAX_LENGTH`: 文本最大长度
  - `BATCH_SIZE`: 批次大小
  - `LEARNING_RATE`: 学习率
  - `NUM_EPOCHS`: 训练轮数
  - `LORA_R`: LoRA秩
  - `LORA_ALPHA`: LoRA缩放因子
  - `LORA_DROPOUT`: LoRA dropout率

2. 开始训练：
```bash
python train.py
```

训练过程中会：
- 自动使用GPU（如果可用）
- 使用wandb记录训练指标
- 保存最佳模型到 `output` 目录

## 模型结构

- 基础模型：Qwen-0.6B-Base
- 微调方法：LoRA
- 目标模块：q_proj, k_proj, v_proj, o_proj
- 训练策略：使用半精度训练（FP16）

## 注意事项

1. 确保有足够的GPU显存（建议至少8GB）
2. 训练数据需要包含足够的正负样本
3. 可以根据实际需求调整模型参数和训练配置
4. 建议使用wandb监控训练过程

## 文件说明

- `train.py`: 训练脚本
- `process_data.py`: 数据处理脚本
- `requirements.txt`: 项目依赖
- `models/`: 预训练模型目录
- `output/`: 训练输出目录 