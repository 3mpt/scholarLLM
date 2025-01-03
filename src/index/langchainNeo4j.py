import getpass
import os

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "123qwezxc"

api_base_url =  "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
model = "internlm2.5-latest"
api_key = "my_key"
from langchain_neo4j import Neo4jGraph

# graph = Neo4jGraph()
enhanced_graph = Neo4jGraph(enhanced_schema=True)
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name=model,          # 使用你的模型
    temperature=0,           # 设置温度等其他参数
    openai_api_base=api_base_url,  # 传递 api_base_url
    openai_api_key=api_key,  # 传递 api_key
)
from operator import add
from typing import Annotated, List

from typing_extensions import TypedDict
# ======================首先，我们将定义 LangGraph 应用程序的 Input、Output 和 Overall 状态。 ==================================

class InputState(TypedDict):
    question: str


class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]


class OutputState(TypedDict):
    answer: str
    steps: List[str]
    cypher_statement: str

# ========第一步是一个简单的步骤，我们验证问题是否与煤矿安全有关。如果没有，我们会通知用户我们无法回答任何其他问题。否则，我们继续进行 Cypher 生成步骤。 ==================================
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
    decision: Literal["coal", "end"] = Field(
        description="关于该问题是否与煤矿安全有关的决定"
    )

# 定义guardrails_chain，组合了聊天模板和结构化输出
guardrails_chain = guardrails_prompt | llm
# guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

# guardrails函数是主要逻辑，用于根据用户问题做出判断
def guardrails(state: InputState) -> OverallState:
    """
    决定问题是否与煤矿安全有关。
    """
    # 调用guardrails_chain，传入问题，返回结构化输出
    guardrails_output = guardrails_chain.invoke({"question": state.get("question")})
    
    # 如果问题与煤矿安全无关，设置database_records为提示信息
    database_records = None
    if guardrails_output.content == "end":
        database_records = "这个问题与煤矿安全无关。因此，我无法回答这个问题。"

    # 返回包含下一步动作、数据库记录和处理步骤的结果
    return {
        "next_action": guardrails_output.content,  # 根据决策返回“coal”或“end”
        "database_records": database_records,  # 如果是“end”，返回相关提示信息
        "steps": ["guardrail"],  # 当前处理步骤
    }

# ================================Few-shot 提示==========================================
# 将自然语言转换为准确的查询具有挑战性。增强此过程的一种方法是提供相关的 
# few-shot 示例来指导 LLM 生成查询。为此，我们将使用 动态选择最相关的示例。
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
examples = [
    {
        "question": "测井人员不得干什么?",
        "query": "MATCH (person:组织与人员 {name: '测井人员'})-[:不得]->(activity:活动与作业) RETURN person.name AS person_name, activity.name AS activity_name",
    },
   
    
]
embed_model = HuggingFaceEmbeddings(
#指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/model/sentence-transformer"
)
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embed_model, Neo4jVector, k=5, input_keys=["question"]
)

# ================================text2cypher==========================================
#接下来，我们实现 Cypher 生成链，也称为 text2cypher。
# 该提示包括增强的图形架构、动态选择的少数样本示例和用户的问题。这种组合允许生成 Cypher 查询以从数据库中检索相关信息。

from langchain_core.output_parsers import StrOutputParser

text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""
            ),
        ),
    ]
)

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()


def generate_cypher(state: OverallState) -> OverallState:
    """
    Generates a cypher statement based on the provided schema and user input
    """
    NL = "\n"
    fewshot_examples = (NL * 2).join(
        [
            f"Question: {el['question']}{NL}Cypher:{el['query']}"
            for el in example_selector.select_examples(
                {"question": state.get("question")}
            )
        ]
    )
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("question"),
            "fewshot_examples": fewshot_examples,
            "schema": enhanced_graph.schema,
        }
    )
    return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

# ================================查询验证==========================================
#下一步是验证生成的 Cypher 语句并确保所有属性值都准确无误。
# 虽然数字和日期通常不需要验证，但电影标题或人名等字符串需要验证。在此示例中，我们将使用基本子句进行验证，但如果需要，可以实现更高级的映射和验证技术。
from typing import List, Optional

validate_cypher_system = """
You are a Cypher expert reviewing a statement written by a junior developer.
"""

validate_cypher_user = f"""
You are a Cypher expert reviewing a statement written by a junior developer.
You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relationship types missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{{schema}}

The question is:
{{question}}

The Cypher statement is:
{{cypher}}

Make sure you don't make any mistakes!"""

# validate_cypher_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             validate_cypher_system,
#         ),
#         (
#             "human",
#             (validate_cypher_user),
#         ),
#     ]
# )
validate_cypher_prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(validate_cypher_user)],
    input_variables=["schema","question","cypher"]
)


class Property(BaseModel):
    """
    Represents a filter condition based on a specific node property in a graph in a Cypher statement.
    """

    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against."
    )


class ValidateCypherOutput(BaseModel):
    """
    Represents the validation result of a Cypher query's output,
    including any errors and applied filters.
    """

    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )


validate_cypher_chain = validate_cypher_prompt | llm

# ================================CypherQueryCorrector==========================================
#LLM 经常难以正确确定生成的 Cypher 语句中的关系方向。
# 由于我们可以访问架构，因此可以使用 CypherQueryCorrector 确定性地更正这些方向。

from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

# Cypher query corrector is experimental
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in enhanced_graph.structured_schema.get("relationships")
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)

# 现在我们可以实现 Cypher 验证步骤。首先，我们使用该方法检测任何语法错误。
# 接下来，我们利用 LLM 来识别潜在问题并提取用于筛选的属性。对于字符串属性，我们使用一个简单的子句针对数据库验证它们。EXPLAINCONTAINS
# 根据验证结果，该过程可以采用以下路径：
# 如果值映射失败，我们将结束对话并通知用户我们无法识别特定属性值（例如，人员或电影标题）。
# 如果发现错误，我们将路由查询以进行更正。
# 如果未检测到问题，我们将继续执行 Cypher 执行步骤。
from neo4j.exceptions import CypherSyntaxError


def validate_cypher(state: OverallState) -> OverallState:
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    errors = []
    mapping_errors = []
    # Check for syntax errors
    try:
        enhanced_graph.query(f"EXPLAIN {state.get('cypher_statement')}")
    except CypherSyntaxError as e:
        errors.append(e.message)
    # Experimental feature for correcting relationship directions
    corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher == state.get("cypher_statement"):
        print("Relationship direction was corrected")
    # Use LLM to find additional potential errors and get the mapping for values
    llm_output = validate_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "schema": enhanced_graph.schema,
            "cypher": state.get("cypher_statement"),
        }
    )
    print(llm_output)
    if llm_output.errors:
        errors.extend(llm_output.errors)
    if llm_output.filters:
        for filter in llm_output.filters:
            # Do mapping only for string values
            if (
                not [
                    prop
                    for prop in enhanced_graph.structured_schema["node_props"][
                        filter.node_label
                    ]
                    if prop["property"] == filter.property_key
                ][0]["type"]
                == "STRING"
            ):
                continue
            mapping = enhanced_graph.query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                {"value": filter.property_value},
            )
            if not mapping:
                print(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
                mapping_errors.append(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
    if mapping_errors:
        next_action = "end"
    elif errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "steps": ["validate_cypher"],
    }
    
# Cypher 更正步骤采用现有的 Cypher 语句、任何已识别的错误和原始问题来生成查询的更正版本。
correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """
            ),
        ),
    ]
)

correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()


def correct_cypher(state: OverallState) -> OverallState:
    """
    Correct the Cypher statement based on the provided errors.
    """
    corrected_cypher = correct_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "errors": state.get("cypher_errors"),
            "cypher": state.get("cypher_statement"),
            "schema": enhanced_graph.schema,
        }
    )

    return {
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "steps": ["correct_cypher"],
    }
# 我们需要添加一个执行给定 Cypher 语句的步骤。如果未返回任何结果，我们应该明确处理这种情况，因为将上下文留空有时会导致 LLM 幻觉。
no_results = "I couldn't find any relevant information in the database"


def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """

    records = enhanced_graph.query(state.get("cypher_statement"))
    return {
        "database_records": records if records else no_results,
        "next_action": "end",
        "steps": ["execute_cypher"],
    }
# 最后一步是生成答案。这涉及将初始问题与数据库输出相结合，以产生相关的回答。
generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
a succinct, definitive answer to the user's question.

Respond as if you are answering the question directly.

Results: {results}
Question: {question}"""
            ),
        ),
    ]
)

generate_final_chain = generate_final_prompt | llm | StrOutputParser()


def generate_final_answer(state: OverallState) -> OutputState:
    """
    Decides if the question is related to movies.
    """
    final_answer = generate_final_chain.invoke(
        {"question": state.get("question"), "results": state.get("database_records")}
    )
    return {"answer": final_answer, "steps": ["generate_final_answer"]}

# 接下来，我们将实现 LangGraph 工作流，从定义条件边函数开始。
def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "coal":
        return "generate_cypher"


def validate_cypher_condition(
    state: OverallState,
) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "correct_cypher":
        return "correct_cypher"
    elif state.get("next_action") == "execute_cypher":
        return "execute_cypher"

# 现在让我们把它们放在一起。
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph

langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(validate_cypher)
langgraph.add_node(correct_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges(
    "guardrails",
    guardrails_condition,
)
langgraph.add_edge("generate_cypher", "validate_cypher")
langgraph.add_conditional_edges(
    "validate_cypher",
    validate_cypher_condition,
)
langgraph.add_edge("execute_cypher", "generate_final_answer")
langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("generate_final_answer", END)

langgraph = langgraph.compile()

# View
# display(Image(langgraph.get_graph().draw_mermaid_png()))
langgraph.invoke({"question": "煤矿测井人员不得干什么?"})
# chain = GraphCypherQAChain.from_llm(
#     graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True
# )

# response = chain.invoke({"query": "煤气层企业应当符合哪些法律法规?"})
# response
# print(response)