import sys
import os

# 获取 `KBQG` 目录的路径
KBQG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 把 `KBQG` 目录添加到 Python 搜索路径
sys.path.append(KBQG_PATH)

# 现在可以正常导入
from model.triple2question import Triple2QuestionModel
from model.randeng_T5 import RandengT5
from model.bart import Bart
from model.QuestionRewriter import QuestionRewriter

from utils.utils import calculate_metrics
import torch
import json
import jieba
from tqdm import tqdm
import csv
import logging
import argparse
import re
import pandas as pd
from bert_score import score
# 设置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    加载测试数据。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_path(path):
    if not path:
        return ""
    
    formatted = path[0][0]  # 取第一个三元组的头实体
    for triple in path:
        formatted += f"的{triple[1]}"  # 追加关系
    
    return formatted

def load_model(model_path, device):
    """
    加载模型。
    """
    model = Bart(device=device)
    # model = RandengT5(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate(model, test_data, rewriter):
    """
    生成问题
    """
    generated_data = []

    # 提取问题和路径
    # questions_list = [item["q"] for item in test_data]
    paths_list = [", ".join([str(triple) for triple in item["path"]]) for item in test_data]
    # print(questions_list, paths_list)
    # # 计算 P, R, F1
    # P, R, F1 = score(questions_list, paths_list, lang="zh", verbose=True)

    for item in tqdm(test_data, desc="生成进度"):
        path = item["path"]
        path_str = ", ".join([str(triple) for triple in path])
        input_text = f"给定知识：{item['path']}，问题的答案是：{item['answer']}，请生成问题"
        # reference_question = entry["q"]

        question = model.generate(input_text)
        
        if question:
            # 判断模型生成的数据是否满足阈值
            # rewritten_question = question if f1_score > 0.8 else rewriter.rewrite_question(path, question)
            # rewritten_question = rewriter.rewrite_question(path, question)
            generated_data.append({
                "输入文本": path,
                "生成问题": question,
                # "参考问题": reference_question,
                # "改写问题": rewritten_question,
                # "F1": f1_score
            })
        else:
            logger.warning("生成失败")
    
    return pd.DataFrame(generated_data)

def main():
    parser = argparse.ArgumentParser(description="生成问题")
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/demo/test_20.json",
        help="测试数据文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../output/model/bart_ans.pth",
        help="模型文件路径",
    )
    parser.add_argument(
        "--output_file", type=str, default="../output/generation/bart_01.csv", help="输出文件路径"
    )
    parser.add_argument(
        "--model_name", type=str, default="fnlp/bart-base-chinese", help="模型名字"
    )

    args = parser.parse_args()

    # 指定设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 加载数据
        test_data = load_data(args.test_data)

        # 加载模型
        model = load_model(args.model_path, device)
        # 初始化改写器
        api_base_url = "http://127.0.0.1:8000/v1/"
        model_name = "models/internlm2_5-7b-chat"
        api_key = "my_key"
        rewriter = QuestionRewriter(api_base_url, model_name, api_key)
        # 生成问题
        generate_result = generate(model, test_data, rewriter)
        generate_result.to_csv(args.output_file, index=False, encoding="utf-8")
        logger.info(f"生成结果已保存到: {args.output_file}")

    except FileNotFoundError as e:
        logger.error(f"文件未找到，请检查路径是否正确。{e}")
    except Exception as e:
        logger.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()
