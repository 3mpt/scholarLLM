import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.bart import Bart
from tqdm import tqdm
import logging
import re
import csv
# 设置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
def load_model(model_path, device):
    """
    加载模型。
    """
    model = Bart(device=device)
    # model = RandengT5(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_data(file_path):
    """
    加载测试数据。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """
    去除文本中的标点符号。
    """
    # 使用正则表达式去除标点符号
    cleaned_text = re.sub(r'[^\w\s]', '', text).replace(" ", "")
    return cleaned_text

# 生成问题的函数
def generate_question(model, test_data):
    result = []
    for entry in tqdm(test_data, desc="生成进度"):
        # 构建输入文本
        path = entry["path"]
        input_text_parts = []
        for subject, predicate, obj in path:
            input_text_parts.append(f"['{subject}', '{predicate}', '{obj}']")
        input_text = f"generate question: [{', '.join(input_text_parts)}]"

        logger.info(f"生成输入文本: {input_text}")
        question = model.generate(input_text)
        if question:
            logger.info(f"生成的问题: {question}")
            question_cleaned = clean_text(question)
            result.append({
                "input_text":input_text,
                "question":question_cleaned
            })
    return result

# 保存结果为 CSV 文件的函数
def save_results_to_csv(results, output_path):
    """
    将生成的问题结果保存为 CSV 文件。
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['input_text', 'generated_question']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
             writer.writerow({
                'input_text': row['input_text'],
                'generated_question': row['question']
            })

# 主函数
if __name__ == "__main__":
    # 加载模型和分词器
    model_path = "output/bart_02_03_18_14.pth"
    data_path = "data/COAL/triples_result.json"
    # 指定设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = load_data(data_path)
    # 加载模型
    model = load_model(model_path, device)
    res = generate_question(model, test_data)
    output_path = "output/example/generated_questions.csv"
    # print(res)
    save_results_to_csv(res, output_path)
    print(f"结果已保存到 {output_path}")