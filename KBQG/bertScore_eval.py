import pandas as pd
from bert_score import score

# 📌 1. 读取 CSV 文件
input_file = "./output/eval/bart_KEGLUE_02_03_18_14.csv"  # 修改为你的 CSV 文件路径
output_file = "./output/eval/bart_KEGLUE_BERTSCORE.csv"  # 输出文件名

df = pd.read_csv(input_file, skiprows=2)

# 📌 2. 计算 BERTScore
refs = df["参考问题"].tolist()  # 参考问题（标注答案）
cands = df["生成问题"].tolist()  # 生成的问题
print(refs)
# 🔥 批量计算 BERTScore
P, R, F1 = score(cands, refs, lang="zh", verbose=True)

# 📌 3. 添加 BERTScore 结果到 CSV
df["BERTScore"] = F1.tolist()  # 将 F1 作为 BERTScore 结果

# 📌 4. 保存结果
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ BERTScore 计算完成，结果已保存至 {output_file}")
