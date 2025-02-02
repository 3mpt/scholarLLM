from model.triple2question import Triple2QuestionModel
from model.randeng_T5 import RandengT5
import torch
# 加载测试数据
import json
from utils import calculate_metrics
with open("./data/test_converted.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
# 指定设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 加载模型
    # model = Triple2QuestionModel(device=device)
    model = RandengT5(device=device)
    model.load_state_dict(torch.load("Randeng_t5_02_02_17_24.pth", map_location=device))
    model.eval()

    # 初始化评估指标
    total_bleu = 0
    total_meteor = 0
    total_rouge_l = 0
    # 评估数据
    # 一条佰 ||| 爷爷 ||| 一条一怔
    for entry in test_data:
        # 构建输入文本
        subject, predicate, obj = entry["path"][0]
        input_text = f"generate question: [['{subject}', '{predicate}', '{obj}']]"
        reference_question = entry["q"]

        print("\n评估输入文本:", input_text)
        question = model.generate(input_text)
        if question:
            print("生成的问题:", question)
            bleu, meteor, rouge_l = calculate_metrics(reference_question, question)
            total_bleu += bleu
            total_meteor += meteor
            total_rouge_l += rouge_l
        else:
            print("生成失败。")

    # 计算平均分数
    num_samples = len(test_data)
    avg_bleu = total_bleu / num_samples
    avg_meteor = total_meteor / num_samples
    avg_rouge_l = total_rouge_l / num_samples

    print(f"\n平均 BLEU 分数: {avg_bleu}")
    print(f"平均 METEOR 分数: {avg_meteor}")
    print(f"平均 ROUGE-L 分数: {avg_rouge_l}")
except FileNotFoundError:
    print("模型文件未找到，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")
