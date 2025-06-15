import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from peft import PeftModel, PeftConfig
from tqdm import tqdm

logging.set_verbosity_error()  # 禁止 transformers 的冗长日志


def load_model_and_tokenizer():
    """加载基础模型和训练好的LoRA模型"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models")  # 基础模型路径
    adapter_path = os.path.join(current_dir, "output")  # LoRA模型路径
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 加载LoRA模型
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16
    )
    
    return model, tokenizer


def format_prompt(input_text):
    """构造改进后的任务提示词"""
    prompt = (
        "任务描述：\n"
        "你需要从给定的社交媒体文本中提取仇恨言论相关的四元组。每个四元组由以下部分组成：\n"
        "评论对象（Target）：帖子的评述对象，如一个人或群体。当无具体目标时设为NULL。\n"
        "论点（Argument）：包含对评论目标的关键论点。\n"
        "目标群体（Targeted Group）：仇恨信息涉及的目标群体之一，包括"'地域'"、"'种族'"、"'性别'"、"'LGBTQ'"、"'其他'"，或无仇恨时设为non-hate。\n"
        "是否仇恨（Hateful）：评论是否包含仇恨，取值为hate或non-hate。\n\n"
        "四元组格式：\n"
        "评论对象 | 论点 | 目标群体 | 是否仇恨 [END]\n"
        "多个四元组用 [SEP] 分隔。\n\n"
        "示例：\n"
        "输入文本：老黑我是真的讨厌，媚黑的还倒贴。\n"
        "输出：老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]\n\n"
        "输入文本：你可真是头蠢驴，这都做不好。\n"
        "输出：你 | 蠢驴 | non-hate | non-hate [END]\n\n"
        "现在请处理以下文本：\n"
        f"输入文本：{input_text}\n"
        "输出："
    )
    return prompt


def generate_response(model, tokenizer, input_text, max_length=512):
    """使用模型生成回复"""
    # 构造提示词
    prompt = format_prompt(input_text)
    
    # 对输入进行编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取模型的实际回复部分
    response = response.split("输出：")[-1].strip()
    
    return response


def process_json_file(model, tokenizer, json_file_path, output_file_path):
    """处理JSON文件中的所有数据"""
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    # 使用tqdm显示进度条
    for item in tqdm(data, desc="处理数据"):
        try:
            # 获取文本内容
            text = item['content']
            # 生成回复
            response = generate_response(model, tokenizer, text)
            # 保存结果
            result = {
                'id': item['id'],
                'content': text,
                'analysis': response
            }
            results.append(result)
            
            # 实时打印当前处理结果
            print(f"\n处理结果 #{item['id']}:")
            print(f"原文: {text}")
            print(f"分析: {response}")
            print("-" * 50)  # 添加分隔线使输出更清晰
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(f"\n处理ID {item['id']} 时发生错误: {e}")
            results.append({
                'id': item['id'],
                'content': text,
                'analysis': error_msg
            })
            print("-" * 50)
    
    # 保存结果到JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！结果已保存到: {output_file_path}")
    return results


def main():
    # 加载模型和分词器
    print("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer()
    print("模型加载完成！")
    
    # 处理test1.json文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "test1.json")
    output_file = os.path.join(current_dir, "test1_results.json")
    
    print(f"\n开始处理文件: {input_file}")
    results = process_json_file(model, tokenizer, input_file, output_file)
    
    # 打印前3个结果作为示例
    print("\n示例结果:")
    for result in results[:3]:
        print(f"\nID: {result['id']}")
        print(f"原文: {result['content']}")
        print(f"分析: {result['analysis']}")


if __name__ == "__main__":
    main()
