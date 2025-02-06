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
        1. 如果问题已经准确且自然地表达了三元组中的主体、关系和客体，请直接返回原问题；
        2. 如果问题与三元组信息匹配度不高或表达不够清晰，请进行改写，使其更自然流畅并包含全部关键信息。

        要求：

        1.保持原问题的核心信息和意思，但可调整表达方式，使其更加自然、流畅。
        2.问题应更符合日常问询语气，可适当增加助词、疑问词等，使其更符合人类提问习惯。
        3.客体应作为问题的答案，问题应由 主体 和 关系 组成。
        4.避免过度重复原问题，但要确保问题仍然表达相同的查询意图。
        5.schema为多个三元组时，应提出多跳问题，前一个三元组的尾节点与下一个三元组的头节点作为连接，不应出现在问题中。
        示例:

        输入:

        schema: [['你好台湾网', '办公地点', '央广新媒体大厦']]
        question: 你好台湾网在哪里办公
        输出: 请问你好台湾网在哪里办公？

        输入:

        schema: [['丰县', '县委书记', '娄海']]
        question: 丰县的县委书记是谁？
        输出: 你知道丰县的县委书记是谁吗？

        输入:

        schema: [['上官瑞谦', '女友', '小渔']]
        question: 请问上官瑞谦的女友是谁？
        输出: 告诉我上官瑞谦的女友是谁？

        输入:

        schema: [['107', '罗马数字', 'CVII']]
        question: 107的罗马数字是什么?
        输出: 请告诉我107的罗马数字?

        输入:

        schema: [['1922年巴西美洲杯', '参赛球队', '巴拉圭、乌拉圭、巴西、阿根廷、智利']]
        question: 请问1922年巴西美洲杯的参赛球队有哪些?
        输出: 有哪些球队参加了1922年巴西美洲杯?

        输入:

        schema: [['浙江升华拜克生物股份有限公司', '经营范围', '马杜霉素'], ['马杜霉素', '注意事项', '泰妙菌素'], ['泰妙菌素', '分子量', '609.8']]
        question: 浙江升华拜克生物股份有限公司的经营范围的注意事项的分子量是多少?
        输出: 浙江升华拜克生物股份有限公司的经营范围的注意事项的分子量是多少?

        输入:

        schema: [['毒舌，傲娇，小豆丁', '中文名', '傲娇'], ['傲娇', '形容', '经常口是心非的女人,经常口是心非的人']]
        question: 毒舌傲娇小豆丁的中文名形容的是什么?
        输出: 谁知道毒舌傲娇小豆丁的中文名形容的是什么?

        输入:

        schema: [['土肥原贤二', '信仰', '军国主义'], ['军国主义', '表现形式', '拿破仑战争']]
        question: 土肥原贤二的信仰表现在哪里?
        输出: 土肥原贤二的信仰是在哪些战争中表现出来的?

        输入:

        schema: [['胡永钢', '毕业院校', '山西大学'], ['山西大学', '学生人数', '15822人']]
        question: 胡永钢的毕业院校有多少学生?
        输出: 胡永钢的毕业院校一共有多少学生?

        输入:

        schema: ['浮山剪纸', '种类', '礼花'], ['礼花', '应用范围', '节日助兴']]
        question: 你知道浮山剪纸的种类应用在哪里吗?
        输出: 浮山剪纸的种类应用在哪个方面?

        任务： 根据给定的 schema 和 question，生成相应的改写问题。
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
