import json
import random
import os
from datetime import datetime

# # 数据集路径
# train_data_path = "./data/NLPCC/train_converted.json"
# test_data_path = "./data/NLPCC/test_converted.json"
# val_data_path = "./data/NLPCC/val_converted.json"
# 数据集路径
train_data_path = "../../data/KGCLUE/train_converted.json"
test_data_path = "../../data/KGCLUE/test_converted.json"
val_data_path = "../../data/KGCLUE/val_converted.json"

# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 读取训练集和测试集数据
with open(train_data_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
with open(val_data_path, "r", encoding="utf-8") as f:
    val_data = json.load(f)
# 输出样本数量
print(f"Training dataset size: {len(train_data)}")
print(f"Test dataset size: {len(test_data)}")
print(f"Val dataset size: {len(val_data)}")

# # 将训练集和测试集合并
# all_data = train_data + test_data +val_data

# # 打乱数据
# random.shuffle(all_data)

# # 计算每个部分的大小
# total_size = len(all_data)
# train_size = int(total_size * train_ratio)
# val_size = int(total_size * val_ratio)
# test_size = int(total_size * test_ratio)

# # 分割数据
# train_data = all_data[:train_size]
# val_data = all_data[train_size : train_size + val_size]
# test_data = all_data[train_size + val_size :]

# # 将训练集、验证集和测试集保存到文件
# with open(train_data_path, "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=4)

# with open(val_data_path, "w", encoding="utf-8") as f:
#     json.dump(val_data, f, ensure_ascii=False, indent=4)

# with open(test_data_path, "w", encoding="utf-8") as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=4)

# 打印保存的文件路径
# print(f"Training data saved to: {train_data_path}")
# print(f"Validation data saved to: {val_data_path}")
# print(f"Test data saved to: {test_data_path}")
