import pandas as pd
from bert_score import score

# 📌 1. 读取 CSV 文件，跳过前两行
input_file = "./output/eval/bart_KEGLUE_02_03_18_14.csv"  # 修改为你的 CSV 文件路径
output_file = "./output/eval/bart_KEGLUE_BERTSCORE.csv"  # 输出文件名
summary_file = "./output/eval/bart_KEGLUE_Summary.csv"  # 平均分数输出文件

# 跳过前两行（使用 skiprows 参数）
df = pd.read_csv(input_file, skiprows=2)

# 📌 2. 计算 BERTScore
refs = df["参考问题"].tolist()  # 参考问题（标注答案）
cands = df["生成问题"].tolist()  # 生成的问题

# 🔥 批量计算 BERTScore
P, R, F1 = score(cands, refs, lang="zh", verbose=True)

# 📌 3. 添加 BERTScore 结果到 CSV
df["BERTScore"] = F1.tolist()  # 将 F1 作为 BERTScore 结果

# 📌 4. 计算平均分数
average_BLEU = df["BLEU"].mean()
average_METEOR = df["METEOR"].mean()
average_ROUGE_L = df["ROUGE-L"].mean()
average_BERTScore = F1.mean()  # 计算BERTScore的平均值

# 📌 5. 创建新的 DataFrame 来存储前两行的平均分数信息
summary = pd.DataFrame({
    "模型名字": ["平均分数"],
    "BLEU": [average_BLEU],
    "METEOR": [average_METEOR],
    "ROUGE-L": [average_ROUGE_L],
    "BERTScore": [average_BERTScore]  # 添加 BERTScore 的平均值
})

# 📌 6. 保存平均分数到新的文件
summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

print(f"✅ 平均分数（包括 BERTScore）已保存至 {summary_file}")
