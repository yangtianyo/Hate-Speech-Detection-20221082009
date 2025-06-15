import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def process_data(input_file, output_file, test_size=0.2, random_state=42):
    """
    处理原始数据，转换为训练所需的格式
    
    Args:
        input_file: 输入数据文件路径
        output_file: 输出数据文件路径
        test_size: 测试集比例
        random_state: 随机种子
    """
    # 读取数据
    # 注意：根据实际数据格式调整读取方式
    df = pd.read_csv(input_file)  # 或 pd.read_json(input_file)
    
    # 数据预处理
    # 1. 清理文本
    df['text'] = df['text'].str.strip()
    
    # 2. 移除空值
    df = df.dropna(subset=['text', 'label'])
    
    # 3. 划分训练集和测试集
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  # 确保标签分布一致
    )
    
    # 4. 转换为HuggingFace数据集格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 5. 保存处理后的数据
    train_dataset.to_json('train.json')
    test_dataset.to_json('test.json')
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"标签分布:\n{train_df['label'].value_counts()}")

if __name__ == "__main__":
    # 使用示例
    process_data(
        input_file="raw_data.csv",  # 替换为实际的数据文件路径
        output_file="processed_data"
    ) 