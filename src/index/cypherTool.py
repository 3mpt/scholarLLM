from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, )
from typing import Dict, List, Any, Optional, Tuple
from langchain.graphs import Neo4jGraph
from logger import logger
from pydantic import Field
from string import Template
from promptConst import SYSTEM_CYPHER_PROMPT
import openai
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
HUMAN_TEMPLATE = "{question}"
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
CYPHER_QA_TEMPLATE = """
您是一名自然语言转换助手，,能够根据问题和结果将它们转换为自然语言，即解释性语句。
除转换为自然语言，请勿回复任何解释或任何其他信息。
您永远不会道歉，并严格根据提供的问题及结构化数据转换为自然语言回答。
当由于缺少会话上下文而无法推断语句时，通知用户，并说明缺少的上下文是什么。
我问你的是中文，请直接用中文查询，不需要翻译为英文
信息部分包含所提供的信息，您必须使用这些信息来构造答案。
我给你的信息就是该问题的答案,请结合我的问题将我的信息转换成自然语言给我，不要出现换行符
信息：
{context}
问题：{question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(input_variables=["context", "question"], prompt=CYPHER_QA_TEMPLATE)

CYPHER_NATURE_TEMPLATE = """
您是一名自然语言转换助手,需要将我给你的结构化数据转变为自然语言
除转换为自然语言，请勿回复任何解释或任何其他信息。要求转化的语言准确流畅！！
您永远不会道歉，并严格根据提供的问题及结构化数据转换为自然语言回答。
当由于缺少会话上下文而无法推断语句时，通知用户，并说明缺少的上下文是什么。
我问你的是中文，请直接用中文查询，不需要翻译为英文
信息部分包含所提供的信息，您必须使用这些信息来构造答案。
不要出现换行符
信息：
{context}

Helpful Answer:"""

CYPHER_NATURE_PROMPT = PromptTemplate(input_variables=["context"], prompt=CYPHER_NATURE_TEMPLATE)


class LLMCypherGraphChain(Chain):
    """Chain that interprets a prompt and executes python code to do math.
    """
    llm: Any
    """LLM wrapper to use."""
    system_prompt: BasePromptTemplate = SYSTEM_CYPHER_PROMPT
    human_prompt: BasePromptTemplate = HUMAN_PROMPT

    input_key: str = "question"  #: :meta private:
    output_key: str = "response"  #: :meta private:

    # graph: Neo4jDatabase
    graph: Neo4jGraph = Field(exclude=True)
    memory: ReadOnlySharedMemory
    return_direct: bool = False

    class Config:
        """Configuration for this pydantic object."""
        extra = 'allow'  # extra = Extra.forbid  # arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str], run_manager: Optional[CallbackManagerForChainRun] = None, ) -> Dict[
        str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        logger.debug(f"Cypher generator inputs: {inputs}")
        # chat_prompt = ChatPromptTemplate.from_messages(
        #     [self.system_prompt] + inputs['chat_history'] + [self.human_prompt])
        # chat_prompt = ChatPromptTemplate.from_messages(
        #     [self.system_prompt] + [self.human_prompt])
        cypher_executor = LLMChain(prompt=SYSTEM_CYPHER_PROMPT, llm=self.llm, callbacks=callbacks)
        start_index = self.graph.get_schema.find("The relationships are the following:")
        relationships_text = self.graph.get_schema[start_index + len("The relationships are the following:"):].strip()

        # print(self.graph.get_structured_schema)
        cypher_statement = cypher_executor({"question": question, "schema": self.graph.get_structured_schema})
        # cypher_statement = cypher_executor({"question": question})
        # print(cypher_statement["text"])
        # cypher_statement = cypher_executor.predict(question=inputs[self.input_key], stop=["Output:"])
        _run_manager.on_text("生成的Cypher语句：", color="green", end="\n\n", verbose=self.verbose)
        cypher_lang = cypher_statement["text"]
        _run_manager.on_text(cypher_statement["text"], color="blue", end="\n\n", verbose=self.verbose)
        intermediate_steps: List = []
        intermediate_steps.append({"query": cypher_statement["text"]})
        # If Cypher statement was not generated due to lack of context
        # if not "MATCH" in cypher_statement:
        #     return {'answer': 'Missing context to create a Cypher statement'}
        # print(intermediate_steps)
        # context是cyhper生成的答案
        context = self.graph.query(cypher_statement["text"])
        # context=[]
        # logger.debug(f"Cypher generator context: {context}")
        if self.return_direct:
            _run_manager.on_text("*" * 100, end="\n", verbose=self.verbose, color="green", )
            _run_manager.on_text("上下文信息:", end="\n", verbose=self.verbose, color="green", )
            _run_manager.on_text(str(question) + str(context), color="blue", end="\n", verbose=self.verbose)
            intermediate_steps.append({"context": context})
            # 问题拼接答案转为自然语言.
            # qa_chain = LLMChain(llm=ChatOpenAI(temperature=0.1), prompt=CYPHER_QA_PROMPT)

            qa_chain = LLMChain(llm=ChatOpenAI(
                model_name="chatglm",
                openai_api_base="http://127.0.0.1:8000/v1",
                openai_api_key="EMPTY",
                streaming=False,
                max_tokens=2048
            ), prompt=CYPHER_QA_PROMPT)
            # print(CYPHER_QA_PROMPT)
            # result = qa_chain({"question": question, "context": context})

            # result1 = result["text"]
            # result1 = result1.replace('\n', '')
            # # final_result = result[qa_chain.output_key]
            # # print(result1)
            # _run_manager.on_text("答案:", end="\n", verbose=self.verbose, color="green")
            # _run_manager.on_text(str(result1), color="blue", end="\n", verbose=self.verbose)
            result1 = ''
            response = {'answer': result1, 'cypher': cypher_lang, 'context': context}

            return {self.output_key: response}
        else:
            _run_manager.on_text("*" * 100, end="\n", verbose=self.verbose, color="green", )
            _run_manager.on_text("结构化信息:", end="\n", verbose=self.verbose, color="green", )
            _run_manager.on_text(str(context), color="blue", end="\n", verbose=self.verbose)
            intermediate_steps.append({"context": context})
            # 答案转为自然语言.
            nature_chain = LLMChain(llm=ChatOpenAI(temperature=0.1), prompt=CYPHER_NATURE_PROMPT)
            result = nature_chain({"context": context})
            nature_result = result[nature_chain.output_key]
            _run_manager.on_text("图谱查询到的结构化数据转化为自然语言:", end="\n", verbose=self.verbose, color="green")
            _run_manager.on_text(str(nature_result), color="blue", end="\n", verbose=self.verbose)
            # 上下文
            _run_manager.on_text("上下文信息:", end="\n", verbose=self.verbose, color="green", )
            _run_manager.on_text(str(question) + str(nature_result), color="blue", end="\n", verbose=self.verbose)
            # 自然语言拼接问题，形成最终答案
            qa_chain = LLMChain(llm=ChatOpenAI(temperature=0.1), prompt=CYPHER_QA_PROMPT)
            result = qa_chain({"question": question, "context": nature_result})

            final_result = result[qa_chain.output_key]

            _run_manager.on_text("答案:", end="\n", verbose=self.verbose, color="green")
            _run_manager.on_text(str(final_result), color="blue", end="\n", verbose=self.verbose)

        # return {'answer': result1,'cypher':cypher_lang}


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    from langchain.chat_models import ErnieBotChat

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    # 文心
    # llm = ErnieBotChat(ernie_client_id = 'GXtqGL36q1lEHw7OLrUC6LKt', ernie_client_secret = 'pngSQ1T53oT553Yaz1wpVN84Ef6NhVm1', model_name = 'ERNIE-Bot-4', temperature =0.1 )
    # gpt3.5

    # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.3,openai_api_key="")
    # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.3,openai_api_key="")
    # glm3
    llm = ChatOpenAI(
        model_name="chatglm",
        openai_api_base="http://127.0.0.1:8000/v1",
        openai_api_key="EMPTY",
        streaming=False,
        max_tokens=2048
    )
    # database = Neo4jDatabase(host="bolt://3.82.241.58:7687",
    #                          user="neo4j", password="intake-roll-openings")
    # database = Neo4jDatabase(host="bolt://localhost:7687", user="neo4j", password="12345678")
    database = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="123qwezxc"
    )
    print(database)
    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database, memory=readonlymemory, return_direct=True,
                                validate_cypher=True)  # return_direct为是否转化为直接拼接问题回答。true 直接返回 ，false转为自然语言再返回

    output = chain.run("氧气浓度分为哪几种？")
    print(output)
    # import pandas as pd
    #
    # # 读取CSV文件
    # file_path = 'data/wenxin.csv'
    # df = pd.read_csv(file_path, encoding='utf-8')
    # print(df)
    # # 对每一行进行处理并将结果写回原始 DataFrame
    # for index, row in df.iterrows():
    #     # 在这里添加你的处理逻辑，例如将处理后的结果写入第二列
    #     output = chain.run(row['问题'])
    #     # df.at[index, 'cypher_zhipu'] = output["cypher"]  # 请替换 '新列名' 为你想要的列名
    #     # 将处理后的结果写入当前行的第二列
    #     # print(row['问题'])
    #     df.at[index, 'cypher'] = output["cypher"]  # 请替换 '新列名' 为你想要的列名
    #     df.at[index, '数据'] = str(output["context"])  # 请替换 '新列名' 为你想要的列名
    #     # df.at[index, '答案'] = output["answer"]  # 请替换 '新列名' 为你想要的列名
    #     df.to_csv(file_path, index=False, encoding='utf-8')
    # 将修改后的 DataFrame 写回原始CSV文件