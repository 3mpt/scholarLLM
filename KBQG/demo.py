import json
import random
import os
from datetime import datetime

train_data_path = "./data/NLPCC/train_converted.json"
test_data_path = "./data/NLPCC/test_converted.json"
val_data_path = "./data/NLPCC/val_converted.json"
# 读取训练集和测试集数据
with open(train_data_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
with open(val_data_path, "r", encoding="utf-8") as f:
    val_data = json.load(f)
print(f"Train dataset size: {len(train_data)}")
print(f"Validation dataset size: {len(val_data)}")
print(f"Test dataset size: {len(test_data)}")
