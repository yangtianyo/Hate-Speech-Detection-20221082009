import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers.trainer import TrainerCallback

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# 模型和分词器配置
MODEL_PATH = "models"
OUTPUT_DIR = "output"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 标签配置
LABEL_NAMES = ["normal", "hate"]  # 标签名称列表


# 设置日志记录
class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 创建日志文件名，包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.json"
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'learning_rate': []
        }

        # 设置日志记录器
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.INFO)

        # 创建文件处理器
        fh = logging.FileHandler(self.log_dir / f"training_{timestamp}.log")
        fh.setLevel(logging.INFO)

        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 添加处理器到日志记录器
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_metrics(self, metrics, step):
        """记录训练指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append((step, value))

        # 保存到JSON文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=4)

        # 记录日志
        self.logger.info(f"Step {step}: {metrics}")

    def plot_metrics(self):
        """绘制训练指标图表"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程指标可视化', fontsize=16)

        # 绘制损失和学习率
        ax1 = axes[0, 0]
        steps, losses = zip(*self.metrics_history['loss'])
        ax1.plot(steps, losses, 'b-', label='损失')
        ax1.set_title('训练损失')
        ax1.set_xlabel('步数')
        ax1.set_ylabel('损失值')
        ax1.legend()

        # 绘制准确率
        ax2 = axes[0, 1]
        steps, accuracies = zip(*self.metrics_history['accuracy'])
        ax2.plot(steps, accuracies, 'g-', label='准确率')
        ax2.set_title('准确率')
        ax2.set_xlabel('步数')
        ax2.set_ylabel('准确率')
        ax2.legend()

        # 绘制F1分数
        ax3 = axes[1, 0]
        steps, f1_scores = zip(*self.metrics_history['f1'])
        ax3.plot(steps, f1_scores, 'r-', label='F1分数')
        ax3.set_title('F1分数')
        ax3.set_xlabel('步数')
        ax3.set_ylabel('F1分数')
        ax3.legend()

        # 绘制精确率和召回率
        ax4 = axes[1, 1]
        steps, precisions = zip(*self.metrics_history['precision'])
        steps, recalls = zip(*self.metrics_history['recall'])
        ax4.plot(steps, precisions, 'c-', label='精确率')
        ax4.plot(steps, recalls, 'm-', label='召回率')
        ax4.set_title('精确率和召回率')
        ax4.set_xlabel('步数')
        ax4.set_ylabel('分数')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png')
        plt.close()


def process_output(output_str):
    """
    处理输出字符串，提取标签信息
    例如: "没爹的黑孩 | 到处扔 | Racism | hate [END]" -> "hate"
    """
    parts = output_str.split(" | ")
    if len(parts) >= 4:
        return parts[-2]  # 返回倒数第二个标签（hate）
    return "normal"  # 如果没有明确的hate标签，返回normal


def prepare_dataset(json_file):
    """
    准备数据集，将json格式转换为训练所需的格式
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    labels = []
    label_map = {"hate": 1, "normal": 0}  # 二分类标签映射

    for item in data:
        texts.append(item["content"])
        label = process_output(item["output"])
        labels.append(label_map.get(label, 0))

    return texts, labels


# 自定义数据集类
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    print(f"使用设备: {'GPU' if use_cuda else 'CPU'}")

    # 初始化日志记录器
    logger = TrainingLogger()
    logger.logger.info("开始训练")
    logger.logger.info(f"使用设备: {'GPU' if use_cuda else 'CPU'}")
    logger.logger.info(f"模型路径: {MODEL_PATH}")
    logger.logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.logger.info(f"批次大小: {BATCH_SIZE}")
    logger.logger.info(f"学习率: {LEARNING_RATE}")
    logger.logger.info(f"训练轮数: {NUM_EPOCHS}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_cuda else torch.float32,  # 根据是否有GPU选择数据类型
        device_map="auto" if use_cuda else None  # 根据是否有GPU选择设备映射
    )

    # 准备LoRA配置
    model = prepare_model_for_kbit_training(model)

    # 精选目标模块 - 核心注意力层 + 两个关键扩展层
    target_modules = [
        # 基础注意力层（保留原有4个）
        "q_proj", "k_proj", "v_proj", "o_proj",

        # 新增两个关键层：
        "gate_proj",  # 门控投影 - 影响信息流动
        "down_proj"  # 下投影层 - 控制维度变换
    ]

    lora_config = LoraConfig(
        r=LORA_R,  # 保持原始秩设置
        lora_alpha=LORA_ALPHA,  # 保持原始alpha
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",

        # 关键优化：分层配置（高层使用更强适配）
        layers_to_transform
        =[19, 20, 21, 22, 23],  # 只改造最后5层
        fan_in_fan_out
        =True  # 提升残差连接适配性
    )
    model = get_peft_model(model, lora_config)

    # 加载并处理数据集
    texts, labels = prepare_dataset("train.json")

    # 创建训练集
    train_dataset = HateSpeechDataset(
        texts,
        labels,
        tokenizer,
        MAX_LENGTH
    )

    # 训练参数 - 使用最基础的配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=use_cuda,  # 根据是否有GPU决定是否使用fp16
        logging_steps=10,
        save_steps=1000,  # 每1000步保存一次
        save_total_limit=2,  # 只保存最新的2个检查点
        remove_unused_columns=False,
        use_cpu=not use_cuda,  # 使用新的参数替代no_cuda
        local_rank=-1,  # 分布式训练设置
        dataloader_num_workers=4,  # 数据加载的线程数
        run_name="hate-speech-detection",  # 设置运行名称
        disable_tqdm=False,  # 显示进度条
        report_to="none",  # 禁用所有报告
        dataloader_pin_memory=use_cuda,  # 根据是否有GPU设置pin_memory
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 添加回调函数来记录指标
    class MetricsCallback(TrainerCallback):
        def __init__(self, logger):
            self.logger = logger

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                metrics = {k: v for k, v in logs.items() if
                           k in ['loss', 'accuracy', 'f1', 'precision', 'recall', 'learning_rate']}
                if metrics:
                    self.logger.log_metrics(metrics, state.global_step)

    trainer.add_callback(MetricsCallback(logger))

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 绘制训练指标图表
    logger.plot_metrics()
    logger.logger.info("训练完成")


if __name__ == "__main__":
    main()