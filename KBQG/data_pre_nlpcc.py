import json
import re

def clean_path(path):
    """
    清理路径中的 ||| 和数字。
    """
    cleaned_path = []
    for item in path:
        cleaned_item = [re.sub(r'\s*\|\|\|\s*\d+', '', part).strip() for part in item]
        cleaned_path.append(cleaned_item)
    return cleaned_path

def convert_format(input_data):
    """
    将原始数据格式转换为目标格式，并清理路径中的 ||| 和数字。
    """
    output_data = []
    for item in input_data:
        # 提取原始数据
        question = item["q"]
        path = item["path"]

        # 清理路径
        cleaned_path = clean_path(path)

        # 构建目标格式
        new_item = {"q": question, "path": cleaned_path}
        output_data.append(new_item)

    return output_data

def main():
    # 加载原始数据
    input_file = "data/NLPCC/test.json"  # 原始数据路径
    output_file = "data/NLPCC/test_converted.json"  # 清理后的数据路径

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 转换数据格式并清理路径
    output_data = convert_format(input_data)

    # 保存清理后的数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"数据已清理并保存到: {output_file}")

if __name__ == "__main__":
    main()