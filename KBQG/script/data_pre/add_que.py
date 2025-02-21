import json

def add_answer_to_path(data):
    # 提取路径中的最后一项作为答案
    answer = data['path'][-1][-1]  # 获取最后一个数组的最后一个元素
    data['answer'] = answer  # 添加到数据中作为 'answer'
    return data

def main():
    # 加载原始数据
    input_file = "../../data/ALL/val.json"  # 原始数据路径
    output_file = "../../data/ALL/val.json"  # 转换后数据路径

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 转换数据格式
    output_data = [add_answer_to_path(item) for item in input_data]

    # 保存转换后的数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"数据已转换并保存到: {output_file}")


if __name__ == "__main__":
    main()
