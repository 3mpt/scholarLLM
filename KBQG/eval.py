from model.triple2question import Triple2QuestionModel
import torch

# 指定设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 加载模型
    model = Triple2QuestionModel(device=device)
    model.load_state_dict(torch.load("triple2question_model.pth", map_location=device))
    model.eval()

    # 评估数据
    eval_data = [
        "generate question: [['辛安镇（黄岛区下辖乡镇）', '东经', '120°06′至120°12′']]",
        "generate question: [['半坡村（云南省曲靖富源县后所镇半坡村）', '平均气温', '18 ℃']]",
        "generate question: [['远日行星', '自转周期', '17小时14分和15小时57分']]",
        "generate question: [['哈药', '职工', '2.01万人']]",
        "generate question: [['国民革命军第17军（闽军李凤翔部组成的第17军）', '改名时间', '1926年9月']]",
        "generate question: [['厦门金龙汽车集团股份有限公司', '证券代码', '600686']]",
        "generate question: [['札幌', '市鸟', '杜鹃鸟']]",
        "generate question: [['Lphant', '支持', 'eDonkey网络']]",
        "generate question: [['库·丘林（凯尔特神话中的英雄）', '人物出处', '凯尔特神话']]",
        "generate question: [['石光荣', '儿女', '石林、石晶、石海']]",
        "generate question: [['大专生', '学制', '三年']]",
        "generate question: [['东马格拉哨所', '所属省份', '内蒙古']]",
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
