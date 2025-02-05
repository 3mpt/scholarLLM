from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
import json
import os
import logging
import pandas as pd
# 设置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# API 配置
api_base_url = "http://127.0.0.1:8000/v1/"
model = "models/internlm2_5-7b-chat"
api_key = "my_key"

# 初始化语言模型
llm = ChatOpenAI(
    model_name=model,
    temperature=0,
    openai_api_base=api_base_url,
    openai_api_key=api_key,
)

# 生成问题系统的提示模板
generate_system = """
给定一个三元组数据 schema 和由其生成的问题 question，请生成一个流畅且自然的改写问题。具体要求如下：

改写后的问题应该尽量模仿示例风格，保留原问题的核心信息和意思。
保证问题的语气自然、流畅，并且符合日常口语表达。
生成的问题应该把客体作为问题的答案，且由主体和关系生成问题。
不需要完全重复原问题，但要保证意思一致，并尽量使问题更具问询性质。
根据上下文，适当变换提问方式，比如使用“你知道”、“请问”等引导词。
示例:

输入:

schema: [['你好台湾网', '办公地点', '央广新媒体大厦']]
question: 你好台湾网在哪里办公
输出: 请问你好台湾网在哪里办公？

输入:

schema: [['丰县', '县委书记', '娄海']]
question: 丰县的县委书记是谁
输出: 你知道丰县的县委书记是谁吗？

任务： 根据给定的 schema 和 question，生成相应的改写问题。
{schema},{question}
"""

# 生成问题的提示模板
generate_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(generate_system)],
    input_variables=["schema", "question"],
)

# 生成问题的输出模型
class GenerateOutput(BaseModel): 
    rewrite: str = Field(..., description="根据示例进行改写")

# 生成问题的链
generate_chain = generate_prompt | llm.with_structured_output(GenerateOutput)

def generate_questions(test_data):
    """
    根据数据生成问题。
    """
    generated_questions = []
    input_texts = test_data["输入文本"].tolist()
    questions = test_data["生成问题"].tolist()
    # 为了将结果添加为新列
    rewritten_questions = []

    for input_text, question in zip(input_texts, questions):
        print(input_text, question)
        input_text = input_text.replace("generate question: ", "")  # 移除 "generate question: "
        rewrite = generate_chain.invoke({"schema": input_text, "question": question}).rewrite
        rewritten_questions.append(rewrite)
    # 将改写问题添加为 DataFrame 新的一列
    test_data["改写问题"] = rewritten_questions
    return test_data

if __name__ == "__main__":
    # 读取 CSV 文件，并将第一列作为索引
    df = pd.read_csv("output/eval/test.csv", index_col=0)
    df.reset_index(drop=True, inplace=True)
    df.index += 1  # 将索引从1开始编号
    df_filtered = df[['输入文本', '生成问题']]
    # 打印结果
    print(df_filtered.head())

    # data_path = "./data/NLPCC/test_converted.json"
    # test_data = load_data(data_path)
    updated_df = generate_questions(df)
    # 保存到原始文件，添加新列 "改写问题"
    updated_df.to_csv("output/eval/bart_KEGLUE_BERTSCORE_with_rewritten.csv", index=False)

    # 打印生成的 DataFrame
    print(updated_df.head())
