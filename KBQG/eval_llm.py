from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import csv
import logging
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API 配置
api_base_url = "http://127.0.0.1:8000/v1/"
model = "models/internlm2_5-7b-chat"
api_key = "my_key"

# 初始化语言模型
llm = ChatOpenAI(
    model_name=model,  # 使用你的模型
    temperature=0,  # 设置温度等其他参数
    openai_api_base=api_base_url,  # 传递 api_base_url
    openai_api_key=api_key,  # 传递 api_key
)

# 生成问题系统的提示模板
generate_system = """
你将接收到一个或多个三元组结构的数据，每个其中包括三部分：主体、关系、客体。
请根据以下三元组生成一个相关的问题语句。
要求：
1. 生成的问题尽可能简短
2. 要基于给定的知识
3. 客体为问题的答案，由主体和关系生成出问题
例如: {{'schema':[['宣肺化痰', '治疗', '外感风寒，痰涎壅肺']],'question':'我想知道宣肺化痰可以治疗什么'}}
    {{'schema': [["顾菁菁","登场作品","一仆二主"], [ "一仆二主", "出品公司","新丽传媒股份有限公司"]],"question":"请问顾菁菁的登场作品是哪个公司出品的？"}}
这是数据
{schema}
"""

# 生成问题的提示模板
generate_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(generate_system)],
    input_variables=["schema"],
)

# 生成问题的输出模型
class GenerateOutput(BaseModel): 
    question: str = Field(..., description="基于给定的三元组生成问题")

# 生成问题的链
generate_chain = generate_prompt | llm.with_structured_output(GenerateOutput)

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

def evaluate(model, test_data):
    """
    评估模型。
    """
    total_bleu = 0
    total_meteor = 0
    total_rouge_l = 0
    evaluation_results = []

    for entry in tqdm(test_data, desc="评估进度"):
        # 构建输入文本
        input_text = entry["path"]
        reference_question = entry["q"]
        logger.info(f"评估输入文本: {input_text}")
        question = model.invoke({"schema": input_text}).question
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
    # 加载数据
    data_path = "./data/NLPCC/test_converted.json"
    test_data = load_data(data_path)
    # 保存结果
    output_path = "output/eval/internlm2_5_nlpcc.csv"

    avg_bleu, avg_meteor, avg_rouge_l, evaluation_results = evaluate(
        generate_chain, test_data
    )
    # 保存评估结果
    save_results(
        output_path,
        'internlm2_5',
        avg_bleu,
        avg_meteor,
        avg_rouge_l,
        evaluation_results,
    )

    # 打印结果保存信息
    logger.info(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    main()