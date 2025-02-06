import sys
import os

# 获取 `KBQG` 目录的路径
KBQG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 把 `KBQG` 目录添加到 Python 搜索路径
sys.path.append(KBQG_PATH)
import torch
import json
from utils.utils import calculate_metrics
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

def clean_text(text):
    """
    去除文本中的标点符号。
    """
    # 使用正则表达式去除标点符号
    cleaned_text = re.sub(r'[^\w\s]', '', text).replace(" ", "")
    return cleaned_text

def evaluate(ques, refs, rewrites):
    """
    评估模型。
    """
    ques_bleu = 0
    ques_meteor = 0
    ques_rouge_l = 0
    ques_BERTScore = 0
    rewrites_bleu = 0
    rewrites_meteor = 0
    rewrites_rouge_l = 0
    rewrites_BERTScore = 0
    evaluation_results = []
    individual_results = []
    ques_processed = [clean_text(ques_text) for ques_text in ques]
    refs_processed = [clean_text(refs_text) for refs_text in refs]
    rewrites_processed = [clean_text(rewrites_text) for rewrites_text in rewrites]
    ques_P, ques_R, ques_F1 = score(ques_processed, refs_processed, lang="zh", verbose=True)
    rewrites_P, rewrites_R, rewrites_F1 = score(rewrites_processed, refs_processed, lang="zh", verbose=True)
    ques_F1_list=ques_F1.tolist()
    rewrites_F1_list=rewrites_F1.tolist()
    for ques_text, refs_text, rewrites_text in tqdm(zip(ques_processed, refs_processed, rewrites_processed), desc="评估进度"):  
        # 分词
        reference_tokens = list(jieba.cut(ques_text))
        generated_tokens = list(jieba.cut(refs_text))
        rewrites_tokens = list(jieba.cut(rewrites_text))

        # 计算 BLEU, METEOR 和 ROUGE-L
        bleu_ques, meteor_ques, rouge_l_ques = calculate_metrics(reference_tokens, generated_tokens)
        bleu_rewrites, meteor_rewrites, rouge_l_rewrites = calculate_metrics(rewrites_tokens, generated_tokens)

        # 累加评估指标
        ques_bleu += bleu_ques
        ques_meteor += meteor_ques
        ques_rouge_l += rouge_l_ques
        rewrites_bleu += bleu_rewrites
        rewrites_meteor += meteor_rewrites
        rewrites_rouge_l += rouge_l_rewrites
        # 将每一行的评估分数记录下来
        individual_results.append({
            "生成问题": ques_text,
            "参考问题": refs_text,
            "改写问题": rewrites_text,
            "生成问题_bleu": bleu_ques,
            "改写问题_bleu": bleu_rewrites,
            "生成问题_meteor": meteor_ques,
            "改写问题_meteor": meteor_rewrites,
            "生成问题_rouge_l": rouge_l_ques,
            "改写问题_rouge_l": rouge_l_rewrites,
        })


    # 计算平均值（可选）
    num_samples = len(ques)
    evaluation_results.append({
        "ques_bleu": ques_bleu / num_samples,
        "ques_meteor": ques_meteor / num_samples,
        "ques_rouge_l": ques_rouge_l / num_samples,
        "ques_BERTScore": ques_F1.mean(),  # 已批量计算的BERTScore
        "rewrites_bleu": rewrites_bleu / num_samples,
        "rewrites_meteor": rewrites_meteor / num_samples,
        "rewrites_rouge_l": rewrites_rouge_l / num_samples,
        "rewrites_BERTScore": rewrites_F1.mean(),  # 已批量计算的BERTScore
    })

    return evaluation_results, individual_results, ques_F1_list, rewrites_F1_list

def main():
    parser = argparse.ArgumentParser(description="评估模型性能")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="../output/generation/nlpcc/bart_rewrite1.csv",
        help="评估数据文件路径",
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="../output/eval/nlpcc/bart_llm1.csv",
         help="输出文件路径"
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="../output/eval/nlpcc/summary1.csv", 
        help="总分文件"
    )
    args = parser.parse_args()

    try:
        # 加载数据
        df = pd.read_csv(args.eval_data)
        
        ques = df["生成问题"].tolist()  # 生成问题
        refs = df["参考问题"].tolist()  # 参考问题（标注答案）
        rewrites = df["改写问题"].tolist()  # 改写问题

        # 评估模型
        evaluation_results, individual_results, ques_F1_list, rewrites_F1_list = evaluate(
            ques, refs, rewrites
        )
        if evaluation_results:
            # 将总评估结果写入 CSV 文件
            df_results = pd.DataFrame(evaluation_results)
            df_results.to_csv(args.summary_file, index=False)
            logger.info(f"评估结果已保存到 {args.summary_file}")
        if individual_results:
            # 每行的评估结果写入文件
            df_results = pd.DataFrame(individual_results)
            df_results["输入文本"] = df["输入文本"]
            df_results["生成问题BERTScore"] = ques_F1_list
            df_results["改写问题BERTScore"] = rewrites_F1_list
            columns_order = ["输入文本"] + [col for col in df_results.columns if col not in ["输入文本"]]
            df_results = df_results[columns_order]
            df_results.to_csv(args.output_file, index=False)
            logger.info(f"评估结果已保存到 {args.output_file}")


    except FileNotFoundError:
        logger.error("文件未找到，请检查路径是否正确。")
    except Exception as e:
        logger.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()
