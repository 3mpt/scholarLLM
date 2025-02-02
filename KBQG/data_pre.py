# 数据预处理
import json


def convert_format(input_data):
    """
    将原始数据格式转换为目标格式。
    """
    output_data = []
    for item in input_data:
        # 提取原始数据
        question = item["question"]
        answer = item["answer"]

        # 解析三元组
        subject, relation, object_ = answer.split(" ||| ")

        # 构建目标格式
        new_item = {"q": question, "path": [[subject, relation, object_]]}
        output_data.append(new_item)

    return output_data


def main():
    # 加载原始数据
    input_file = "data/test.json"  # 原始数据路径
    output_file = "data/test_converted.json"  # 转换后数据路径

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 转换数据格式
    output_data = convert_format(input_data)

    # 保存转换后的数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"数据已转换并保存到: {output_file}")


if __name__ == "__main__":
    main()
