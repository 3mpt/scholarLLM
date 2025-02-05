from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field


class GenerateOutput(BaseModel):
    rewrite: str = Field(..., description="根据示例进行改写")


class QuestionRewriter:
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
        给定一个三元组数据 schema 和由其生成的问题 question，请生成一个流畅且自然的改写问题。具体要求如下：
        改写后的问题应该尽量模仿示例风格，保留原问题的核心信息和意思。
        保证问题的语气自然、流畅，并且符合日常口语表达。
        生成的问题应该把客体作为问题的答案，且由主体和关系生成问题。
        不需要完全重复原问题，但要保证意思一致，并尽量使问题更具问询性质。
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

        schema: [['The Bravery（大无畏乐团）', '建馆时间', '2003年']]
        question: 你知道TheBravery是什么时候建的吗?
        输出: TheBravery的建馆时间是什么时候?

        输入:

        schema: [['Tony Lee', '个人荣誉', '“2014最具价值CTO”奖']]
        question: 你知道TonyLee获得过什么荣誉吗?
        输出: TonyLee有什么荣誉?

        输入:

        schema: [['长江武汉航道局', '管辖', '715.2公里']]
        question: 长江武汉航道局的管辖范围是什么?
        输出: 你知道长江武汉航道局管多少个地方吗?

        输入:

        schema: [['南亚运动会', '举办方', '南亚地区合作组织']]
        question: 南亚运动会的举办方是哪里?
        输出: 你知道南亚运动会是由谁举办的吗?

        输入:

        schema: [['千佛寺石窟', '最佳旅游时间', '四季皆宜']]
        question: 千佛寺石窟的最佳旅游时间是什么时候?
        输出: 你知道什么时候去千佛寺石窟旅游比较好吗?

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
