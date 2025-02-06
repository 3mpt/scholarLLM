#  数据集路径
NLPCC_train_data_path = "../../data/NLPCC/train_converted.json"
NLPCC_test_data_path = "../../data/NLPCC/test_converted.json"
NLPCC_val_data_path = "../../data/NLPCC/val_converted.json"
# 数据集路径
KGCLUE_train_data_path = "../../data/KGCLUE/train_converted.json"
KGCLUE_test_data_path = "../../data/KGCLUE/test_converted.json"
KGCLUE_val_data_path = "../../data/KGCLUE/val_converted.json"

train_data_path = "../../data/ALL/train.json"
test_data_path = "../../data/ALL/test.json"
val_data_path = "../../data/ALL/val.json"

import json
import random
def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    """保存 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def merge_datasets(file_paths):
    """合并多个数据集"""
    merged_dataset = []
    for file_path in file_paths:
        dataset = load_json(file_path)
        shuffled_dataset = shuffle_dataset(dataset)  # 打乱数据
        merged_dataset.extend(shuffled_dataset)  # 直接扩展列表
    return merged_dataset

def shuffle_dataset(dataset):
    """打乱数据集顺序"""
    print('@')
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    train_data = merge_datasets([NLPCC_train_data_path, KGCLUE_train_data_path])
    test_data = merge_datasets([NLPCC_test_data_path, KGCLUE_test_data_path])
    val_data = merge_datasets([NLPCC_val_data_path, KGCLUE_val_data_path])
    save_json(shuffle_dataset(train_data), train_data_path)
    save_json(shuffle_dataset(test_data), test_data_path)
    save_json(shuffle_dataset(val_data), val_data_path)
    print(len(train_data),len(test_data),len(val_data))
