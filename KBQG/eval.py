from model.triple2question import Triple2QuestionModel
from model.randeng_T5 import RandengT5
import torch

# 指定设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 加载模型
    # model = Triple2QuestionModel(device=device)
    model = RandengT5(device=device)
    model.load_state_dict(torch.load("Randeng_t5_02_02_17_24.pth", map_location=device))
    model.eval()

    # 评估数据
    # 一条佰 ||| 爷爷 ||| 一条一怔
    eval_data = [
       "generate question: [['一条佰', '爷爷', '一条一怔']]",
       "generate question: [['巴淡市', '陆地面积', '1040平方公里']]",
       "generate question: [['火葫芦', '冷却时间', '15s']]",
       "generate question: [['教育方法', '实现', '教育目的，完成教育任务']]",
       "generate question: [['古窑', '展馆名称', '景德镇古窑民俗博览区']]",

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
