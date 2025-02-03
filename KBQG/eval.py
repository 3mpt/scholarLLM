from model.triple2question import Triple2QuestionModel
from model.randeng_T5 import RandengT5
from model.bart import Bart
import torch
import json
from utils import calculate_metrics
import jieba
from tqdm import tqdm
import csv
import logging
import argparse
import os
import re

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


def load_model(model_path, device):
    """
    加载模型。
    """
    model = Bart(device=device)
    # model = RandengT5(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def clean_text(text):
    """
    去除文本中的标点符号。
    """
    # 使用正则表达式去除标点符号
    cleaned_text = re.sub(r'[^\w\s]', '', text).replace(" ", "")
    return cleaned_text

def evaluate(model, test_data, device):
    """
    评估模型。
    """
    total_bleu = 0
    total_meteor = 0
    total_rouge_l = 0
    evaluation_results = []

    for entry in tqdm(test_data, desc="评估进度"):
        # 构建输入文本
        path = entry["path"]
        input_text_parts = []
        for subject, predicate, obj in path:
            input_text_parts.append(f"['{subject}', '{predicate}', '{obj}']")
        input_text = f"generate question: [{', '.join(input_text_parts)}]"
        reference_question = entry["q"]

        logger.info(f"评估输入文本: {input_text}")
        question = model.generate(input_text)
        if question:
            logger.info(f"生成的问题: {question}")
            # 去除标点符号
            reference_question_cleaned = clean_text(reference_question)
            question_cleaned = clean_text(question)
            # 分词
            reference_tokens = list(jieba.cut(reference_question_cleaned))
            generated_tokens = list(jieba.cut(question_cleaned))
            bleu, meteor, rouge_l = calculate_metrics(
                reference_tokens, generated_tokens
            )
            total_bleu += bleu
            total_meteor += meteor
            total_rouge_l += rouge_l

            # 存储评估结果
            evaluation_results.append(
                {
                    "input_text": input_text,
                    "generated_question": question_cleaned,
                    "reference_question": reference_question_cleaned,
                    "bleu": bleu,
                    "meteor": meteor,
                    "rouge_l": rouge_l,
                }
            )
        else:
            logger.warning("生成失败。")

    # 计算平均分数
    num_samples = len(test_data)
    avg_bleu = total_bleu / num_samples
    avg_meteor = total_meteor / num_samples
    avg_rouge_l = total_rouge_l / num_samples

    logger.info(f"平均 BLEU 分数: {avg_bleu}")
    logger.info(f"平均 METEOR 分数: {avg_meteor}")
    logger.info(f"平均 ROUGE-L 分数: {avg_rouge_l}")

    return avg_bleu, avg_meteor, avg_rouge_l, evaluation_results


def save_results(
    output_file,
    model_name,
    avg_bleu,
    avg_meteor,
    avg_rouge_l,
    evaluation_results,
    append=False,
):
    """
    保存评估结果到 CSV 文件。
    """
    mode = "a" if append else "w"
    with open(output_file, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not append:
            # 写入表头
            writer.writerow(
                ["模型名字", "平均 BLEU 分数", "平均 METEOR 分数", "平均 ROUGE-L 分数"]
            )
        # 写入评估结果
        writer.writerow([model_name, avg_bleu, avg_meteor, avg_rouge_l])

        # 写入每个样本的详细评估结果
        writer.writerow(
            ["", "输入文本", "生成问题", "参考问题", "BLEU", "METEOR", "ROUGE-L"]
        )
        for result in evaluation_results:
            writer.writerow(
                [
                    "",
                    result["input_text"],
                    result["generated_question"],
                    result["reference_question"],
                    result["bleu"],
                    result["meteor"],
                    result["rouge_l"],
                ]
            )


def main():
    parser = argparse.ArgumentParser(description="评估模型性能")
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/NLPCC/test_converted.json",
        help="测试数据文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/bart_NLPCC_02_03_14_54.pth",
        help="模型文件路径",
    )
    parser.add_argument(
        "--output_file", type=str, default="output/evaluation_results_new_rouge.csv", help="输出文件路径"
    )
    parser.add_argument(
        "--model_name", type=str, default="bart_NLPCC_02_03_14_54", help="模型名字"
    )
    parser.add_argument("--append", action="store_true", help="是否追加结果到现有文件")
    args = parser.parse_args()

    # 指定设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 加载数据
        test_data = load_data(args.test_data)

        # 加载模型
        model = load_model(args.model_path, device)

        # 评估模型
        avg_bleu, avg_meteor, avg_rouge_l, evaluation_results = evaluate(
            model, test_data, device
        )

        # 保存评估结果
        save_results(
            args.output_file,
            args.model_name,
            avg_bleu,
            avg_meteor,
            avg_rouge_l,
            evaluation_results,
            args.append,
        )

        logger.info(f"评估结果已保存到: {args.output_file}")

    except FileNotFoundError:
        logger.error("文件未找到，请检查路径是否正确。")
    except Exception as e:
        logger.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()
