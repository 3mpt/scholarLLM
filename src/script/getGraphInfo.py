import os
from langchain_neo4j import Neo4jGraph
import json

# 设置 Neo4j 连接参数
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "123qwezxc"

# 初始化 Neo4j 图谱
enhanced_graph = Neo4jGraph()

# 定义转换函数
def convert_triples(triples):
    """
    将三元组数据转换为所需的格式。
    
    :param triples: 三元组数据列表，每个三元组包含 'a', 'b', 'r' 字段
    :return: 转换后的结果列表，每个元素包含 'path' 字段
    """
    result = []
    for triple in triples:
        path = []
        a = triple['a']
        b = triple['b']
        r = triple['r']
        
        # 构建路径
        path.append([a, r, b])
        
        # 将路径添加到结果中
        result.append({
            "path": path
        })
    
    return result

# 定义查询语句
query = """
MATCH (a)-[r]->(b)
RETURN a.name as a, type(r) as r , b.name as b LIMIT 10000
"""  # 查询图谱中所有节点，并限制结果数量为 10

# 执行查询
result = enhanced_graph.query(query)

# 转换结果
res = convert_triples(result)

# 将结果写入 JSON 文件
with open('../../KBQG/data/COAL/triples_result.json', 'w', encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=4)

# 打印结果
print(json.dumps(res, ensure_ascii=False, indent=4))