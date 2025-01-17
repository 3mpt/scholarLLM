import os
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "123qwezxc"
api_base_url = "http://127.0.0.1:8000/v1/"
model = "models/internlm2_5-7b-chat"
api_key = "my_key"
enhanced_graph = Neo4jGraph()
llm = ChatOpenAI(
    model_name=model,  # 使用你的模型
    temperature=0,  # 设置温度等其他参数
    openai_api_base=api_base_url,  # 传递 api_base_url
    openai_api_key=api_key,  # 传递 api_key
)
generate_system = """
你将接收到一个四元组结构的数据，其中包括四部分：起始节点（start）、关系类型（type）、开始节点的标签labels(a)、结束节点的标签labels(b)。
请根据以下四元组为每个四元组生成一个相关的问答语句和一个对应的 Cypher 查询语句。
要求：
1. 问题应基于四元组中的数据，问题的答案应是结束节点（end）,问题中不要出现结束节点,而是出现结束节点的label；
2. 生成的 Cypher 查询应基于四元组的结构，查询的目的是验证该关系。

例如:{{
  "start": "测井人员",
  "type": "不得",
  "labels(a)":"组织与人员",
  "labels(b)":"活动与作业"
  }}
"question": "测井人员不得参与什么活动与作业?",
"cypher": "MATCH (person:组织与人员 {{name: '测井人员'}})-[:不得]->(activity:活动与作业) RETURN person.name AS person_name, activity.name AS activity_name"
输出格式:
question: 
cypher: 
结构
{schema}
"""
generate_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(generate_system)],
    input_variables=["schema"],
)


class GenerateOutput(BaseModel):
    question: str = Field(..., description="生成与煤矿安全有关的问题")
    cypher: str = Field(..., description="Cypher 查询语句，基于输入的三元组生成")


generate_chain = generate_prompt | llm.with_structured_output(GenerateOutput)
# 获取所有数据
query = """
MATCH (a)-[r]->(b)
RETURN a.name as start, type(r) as type , labels(a),labels(b) LIMIT 1000
"""  # 查询图谱中所有节点
result = enhanced_graph.query(query)
# 保存数据到列表中
data = []
import csv

# 打开文件并以追加模式写入
with open("output1.csv", mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # 如果是第一次写入，添加表头
    if file.tell() == 0:  # 如果文件为空
        writer.writerow(["五元组信息", "问题", "生成的语句", "查询结果"])

    # 打印并追加结果
    for record in result:
        try:
            output = generate_chain.invoke({"schema": record})  # 生成 Cypher 查询语句

            # 执行查询并捕获结果
            query_result = enhanced_graph.query(output.cypher)

            # 如果查询结果为 false 或空，则设为 False
            if query_result is None or query_result == False:
                query_result = False

            # 打印并写入文件
            writer.writerow([record, output.question, output.cypher, query_result])
            print(record, "\n", output.cypher, query_result)

        except Exception as e:
            # 如果捕获到异常，打印错误信息，并设置 query_result 为 False
            print(f"Error processing record {record}: {e}")
            query_result = False

            # 写入错误信息
            writer.writerow(
                [
                    record,
                    "Error generating question",
                    "Error generating Cypher",
                    query_result,
                ]
            )

            # 打印输出错误信息
            print(record, "\n", "Cypher query failed", query_result)
