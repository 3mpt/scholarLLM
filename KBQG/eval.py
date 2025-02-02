from model.triple2question import Triple2QuestionModel
import torch

# 指定设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 加载模型
    model = Triple2QuestionModel(device=device)
    model.load_state_dict(torch.load("Randeng_t5.pth", map_location=device))
    model.eval()

    # 评估数据
    eval_data = [
        'generate question: [["Sapporo", "City Bird", "Cuckoo"]]',
        'generate question: [["Junior College Student", "Duration of Study", "Three Years"]]',
    ]

    # 生成问题
    for input_text in eval_data:
        print("\n评估输入文本:", input_text)
        question = model.generate(input_text)
        if question:
            print("生成的问题:", question)
        else:
            print("生成失败。")

except FileNotFoundError:
    print("模型文件未找到，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")
