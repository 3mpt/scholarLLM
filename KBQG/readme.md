# KBQG

基于知识库的问题生成（Knowledge-Based Question Generation，KBQG）是问题生成领域的重要分支。它以知识库或查询子图作为输入，生成相关的自然语言问题。与一般的问题生成任务相比，KBQG 的输入是从知识库中获取的事实三元组，通常表示为<s, p, o>

中文 KBQG 的项目开源的少，连论文又没有几个是做的中文，中英文语法句法结构差距大不适合用英文模型去做问题生成，所以就自己写了一个中文 KBQG 的项目。

## 数据集

数据集使用了中文通用领域 KBQG 数据集

KGCLUE: <https://github.com/CLUEbenchmark/KgCLUE>

NLPCC-MH: <https://github.com/wavewangyue/NLPCC-MH>

## 模型

选取了一些中文能力比较好的模型
bart
randeng_T5

TODO:

后续要把这一块融入LLM结合
使用大语言模型0-shot 进行KBQG任务