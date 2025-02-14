from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field


class GenerateOutput(BaseModel):
    score: str = Field(..., description="根据示例进行评分")


class LLMeval:
    def __init__(self, api_base_url: str, model: str, api_key: str):
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=0,
            openai_api_base=api_base_url,
            openai_api_key=api_key,
        )

        self.prompt_template = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template(self._get_generate_system())],
            input_variables=["schema", "question"],
        )

        self.chain = self.prompt_template | self.llm.with_structured_output(GenerateOutput)

    def _get_generate_system(self):
        return """
        任务描述：
        给定一组三元组数据 schema 和由其生成的问题 question，
        请先判断该问题是否准确表达了 schema 的关键信息：
        如果问题已经准确且自然地表达了三元组中的主体、关系和客体，请直接返回 score
        示例:
        输入:
        schema: [['Project Glass', '售价', '1500美元（开发者版）']]
        question: 你知道ProjectGlass的售价是多少吗?
        输出: 0.9
        {schema},{question}
        """

    def rewrite_question(self, schema: list, question: str) -> str:
        result = self.chain.invoke({"schema": schema, "question": question})
        return result.rewrite


# # 使用示例
# api_base_url = "http://127.0.0.1:8000/v1/"
# model = "models/internlm2_5-7b-chat"
# api_key = "my_key"

# rewriter = QuestionRewriter(api_base_url, model, api_key)
# schema = [['107', '罗马数字', 'CVII']]
# question = "107的罗马数字是什么?"
# rewritten_question = rewriter.rewrite_question(schema, question)

# print(rewritten_question)
