# from bert_score import score
# questions_list=['论柏拉图对话的丛书叫什么','你知道ProjectGlass的售价是多少吗']
# paths_list= ['论柏拉图对话的丛书名','Project Glass的售价']
# P, R, F1 = score(questions_list, paths_list, lang="zh", verbose=True)
# print(F1)
import pandas as pd
import ast
from collections import OrderedDict
def check_jump_type(arr):
    print(ast.literal_eval(arr), len(ast.literal_eval(arr)))
    return "单跳" if len(ast.literal_eval(arr)) == 1 else "多跳"
def flatten_and_deduplicate(refs_text):
    # 将字符串转换为列表
    triplets = ast.literal_eval(refs_text)
    
    # 扁平化并去重且保持顺序
    flat_list = [item for triplet in triplets for item in triplet]
    unique_elements = list(OrderedDict.fromkeys(flat_list))  # 保持顺序并去重

    # 转换为字符串，元素之间用空格分隔
    result_string = "".join(unique_elements)
    return result_string
df = pd.read_csv('../../output/generation/ALL/bart_18_23.csv')
# 假设 df 是你的原始 DataFrame
# df = pd.DataFrame({
#     "输入文本": [[['内田真礼', '血型', 'A型']], [['香取茜', '丈夫', '大贺真哉'], ['大贺真哉', '登场作品', '名侦探柯南'], ['名侦探柯南', '揭载号', '1994年第5号（1994年1月5日发行）']], [['蒲棒', '用量', '外用：捣敷']]]  # 示例数据，列表类型
# })

# 创建 df_results 并添加新列
df_results=pd.DataFrame()
df_results["输入文本"] = df["输入文本"]
df_results["跳数类别"] = df["输入文本"].apply(check_jump_type)
df_results["扁平化"] = df["输入文本"].apply(flatten_and_deduplicate)
# 输出结果
print(df_results.head())
