from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

os.environ["OPENAI_API_KEY"] = "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"

def load_markdown_documents(dir, split_mode='single'):
    docs = [];
    files = os.listdir(dir)
    # 读取目标目录下的所有文件
    for file in files:
        if file.endswith('.md'):
            # mode=element 才会进行分割, 否则直接调用 load_and_split 也是不起作用的
            documents = UnstructuredMarkdownLoader('./db/' + file, mode=split_mode).load_and_split()
            docs.extend(documents)
        # 判断是否是文件夹
        elif os.path.isdir(file):
            documents = load_markdown_documents(file)
            docs.extend(documents)
    return docs

def init_milvus():
    from langchain.vectorstores.milvus import Milvus
    # 创建 vector db
    vector_db = Milvus.from_documents(
        documents=load_markdown_documents('./db', split_mode='elements'),
        embedding=OpenAIEmbeddings(),
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )
    return vector_db

def drop_milvus():
    from pymilvus import Milvus
    db = Milvus(connection_args={"host": "127.0.0.1", "port": "19530"}, collection_name="LangChainCollection")
    db.drop_collection("LangChainCollection")

# TODO 校验是否已存在相同条目，重复调用会导致重复插入
# TODO drop old collection?
# TODO 检验递归调用是否成功

# drop_milvus()

from langchain.vectorstores.milvus import Milvus

# 重新加载并初始化 vector db
# vector_db = init_milvus()
# 载入已有数据
vector_db = Milvus(
    connection_args={"host": "127.0.0.1", "port": "19530"}, 
    collection_name="LangChainCollection",
    embedding_function=OpenAIEmbeddings(),
)

query = "2022年11月7日干了什么？"
result = vector_db.similarity_search(query, 2)
print("vector db result:", result)

chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
ans = chain.run(input_documents=result, question=query)
print("chain result:", ans)
