import os

from langchain_openai import ChatOpenAI
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "123qwezxc"

api_base_url =  "http://127.0.0.1:8000/v1/"
model = "models/internlm2_5-7b-chat"
api_key = "my_key"
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
# graph = Neo4jGraph()
# enhanced_graph = Neo4jGraph(enhanced_schema=True)

llm = ChatOpenAI(
    model_name=model,          # 使用你的模型
    temperature=0,           # 设置温度等其他参数
    openai_api_base=api_base_url,  # 传递 api_base_url
    openai_api_key=api_key,  # 传递 api_key
)
# chain = GraphCypherQAChain.from_llm(
#     graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True
# )
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

# 定义系统消息，描述智能助理的行为规则
guardrails_system = f"""
作为一名智能助理，你的主要目标是决定一个给定的问题是否与煤矿安全有关。
如果问题与煤矿安全有关，则输出"coal"。否则，输出"end"。
为了做出这个决定，评估问题的内容，并确定它是否涉及任何安全管理、法律法规、设备与工具、活动与作业、组织与人员、环境与条件，
或相关主题。仅提供指定的输出："coal" 或 "end"。
用户提问：
{{question}}
"""


# 创建聊天模板，系统消息包含安全相关的决策逻辑，用户问题在"human"消息中
guardrails_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(guardrails_system)],
    input_variables=["question"]
)

# 定义GuardrailsOutput类，用于存储模型输出的结果
class GuardrailsOutput(BaseModel):
    # 只有两个可能的输出：“coal”或“end”
    content: Literal["coal", "end"] = Field( 
        description="关于该问题是否与煤矿安全有关的决定"
    )
# parser = PydanticOutputParser(pydantic_object=PlanList)
# 定义guardrails_chain，组合了聊天模板和结构化输出
# question = f"测井人员不得干什么?"
guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)
# guardrails_chain = guardrails_prompt | llm
guardrails_output = guardrails_chain.invoke({"question": "测井人员不得干什么?"})
# response = chain.invoke({"query": "测井人员不得干什么?"})
# response
print(guardrails_output)