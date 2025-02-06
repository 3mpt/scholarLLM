# import json
# import random
# import os
# from datetime import datetime

# train_data_path = "./data/NLPCC/train_converted.json"
# test_data_path = "./data/NLPCC/test_converted.json"
# val_data_path = "./data/NLPCC/val_converted.json"
# # 读取训练集和测试集数据
# with open(train_data_path, "r", encoding="utf-8") as f:
#     train_data = json.load(f)

# with open(test_data_path, "r", encoding="utf-8") as f:
#     test_data = json.load(f)
# with open(val_data_path, "r", encoding="utf-8") as f:
#     val_data = json.load(f)
# print(f"Train dataset size: {len(train_data)}")
# print(f"Validation dataset size: {len(val_data)}")
# print(f"Test dataset size: {len(test_data)}")
# path = [['长江武汉航道局', '管辖', '715.2公里'],['丘堤', '儿媳', '籍虹']]
# path_str = ", ".join([str(triple) for triple in path])
# print(path_str)
import json

# 你的 JSON 数据（如果是文件，使用 json.load 读取）
test_data = [
    {
        "q": "请问王永标的毕业院校有多少个博士点？",
        "path": [
            ["王永标", "毕业院校", "中国地质大学"],
            ["中国地质大学", "博士点", "66个"]
        ]
    },
    {
        "q": "请问开发上海国经网络科技有限公司的合作伙伴这个软件的公司是什么？",
        "path": [
            ["上海国经网络科技有限公司", "合作伙伴", "必应"],
            ["必应", "开发公司", "微软公司"]
        ]
    },
    {
        "q": "你知道纪念那份死去的爱的作者的本义吗？",
        "path": [
            ["纪念那份死去的爱", "作者", "爱屋及乌"],
            ["爱屋及乌", "本义", "爱房子,也爱那房顶上的乌鸦。"]
        ]
    },
    {
        "q": "谁知道浙江九龙山国家级自然保护区的保护对象的名字？",
        "path": [
            ["浙江九龙山国家级自然保护区", "保护对象", "黄腹角雉"],
            ["黄腹角雉", "定名人", "Geoffroy St. Hilaire"]
        ]
    }
]

# 分别提取 `q` 和 `path`（将 `path` 转换为字符串）
questions = [item["q"] for item in test_data]
paths = [", ".join([str(triple) for triple in item["path"]]) for item in test_data]

# 打印结果
print("Questions:", questions)
print("Paths:", paths)
