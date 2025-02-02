from rouge import Rouge
import jieba

# 示例输入
reference = "宁财神的好友的妻子叫什么?"
candidate = "宁财神的好友的妻子叫啥?"

# 分词
reference_tokens = " ".join(list(jieba.cut(reference)))
candidate_tokens = " ".join(list(jieba.cut(candidate)))

# 打印分词结果
print("Reference Tokens:", reference_tokens)
print("Candidate Tokens:", candidate_tokens)

# 初始化 ROUGE 计算器
rouge = Rouge()

# 计算 ROUGE-L 分数
scores = rouge.calc_score([candidate_tokens], [reference_tokens])

# 打印 ROUGE-L 分数
print("ROUGE-L Score:", scores)
