from string import Template
from langchain import PromptTemplate
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, )
# 从文件中读取模板
with open('./prompt/examples.txt', 'r', encoding='utf-8') as file:
    # cypher 示例
    examples = file.read()
with open('./prompt/systemTemplate.txt', 'r', encoding='utf-8') as file:
    systemTemplate = file.read()
systemTemplate1 = Template(systemTemplate)
SYSTEM_TEMPLATE = systemTemplate1.substitute(examples=examples)
SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate(input_variables=["question","schema"], prompt=SYSTEM_TEMPLATE)
