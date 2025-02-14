
from utils.utils import calculate_metrics
import jieba
import ast
from collections import OrderedDict

def flatten_and_deduplicate(refs_text):
    # 将字符串转换为列表
    triplets = ast.literal_eval(refs_text)
    
    # 扁平化并去重且保持顺序
    flat_list = [item for triplet in triplets for item in triplet]
    unique_elements = list(OrderedDict.fromkeys(flat_list))  # 保持顺序并去重

    # 转换为字符串，元素之间用空格分隔
    result_string = "".join(unique_elements)
    return result_string

ques_text = "手机吊带的外文名使用了什么材料"
refs_text = "[['手机吊带', '外文名', '手机绳'], ['手机绳', '主要材料', 'poly']]"
refs_text = flatten_and_deduplicate(refs_text)
print(refs_text)
rewrites_text = '请问手机吊带的外文名是手机绳它主要使用了哪种材料'
reference_tokens = list(jieba.cut(ques_text))
generated_tokens = list(jieba.cut(refs_text))
rewrites_tokens = list(jieba.cut(rewrites_text))
print(reference_tokens, generated_tokens, rewrites_tokens)
# 计算 BLEU, METEOR 和 ROUGE-L
bleu_ques, meteor_ques, rouge_l_ques = calculate_metrics(reference_tokens, generated_tokens)
bleu_rewrites, meteor_rewrites, rouge_l_rewrites = calculate_metrics(rewrites_tokens, generated_tokens)
print(bleu_ques, meteor_ques, rouge_l_ques)
print(bleu_rewrites, meteor_rewrites, rouge_l_rewrites)