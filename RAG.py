# -*- coding: utf-8 -*-
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os


model_local_path=r"D:\Llama3\RAG\m3e\m3e-base"

model_kwargs = {
    'device': 'cuda'
}
encode_kwargs = {'normalize_embeddings': True}
model = HuggingFaceBgeEmbeddings(
    model_name=model_local_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

#D:\Llama3\RAG\testzh.txt
def RAG_Embed(text, persist_directory):
    """
    将文本嵌入向量数据库
    
    参数:
    text (str): 要嵌入的文本文件路径
    persist_directory (str): 持久化向量数据库的目录
    
    返回:
    vectordb (Chroma): 嵌入后的向量数据库
    """
   
    batch_size=5000
    #shutil.rmtree(persist_directory)
    loader = TextLoader(text, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=64, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    vectordb = None
    for i in range(0, len(documents), batch_size) :
        batch_documents = documents[i:i+batch_size]
        if i == 0:
            vectordb = Chroma.from_documents(batch_documents, model, persist_directory=persist_directory)
        else:
            vectordb.add_documents(batch_documents)
    return vectordb


def RAG_Embed_Knowledgebase(text, persist_directory):
    """
    将知识库文本嵌入向量数据库,如果数据库已存在则直接加载
    
    参数:
    text (str): 要嵌入的知识库文本文件路径
    persist_directory (str): 持久化向量数据库的目录
    
    返回:
    vectordb (Chroma): 嵌入后的向量数据库
    """
    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory,  embedding_function=model)
        return vectordb
    return RAG_Embed(text, persist_directory)


def RAG_Embed_History(text, persist_directory, history):
    """
    将对话历史写入文本文件并嵌入向量数据库
    
    参数:
    text (str): 要写入的对话历史文本文件路径
    persist_directory (str): 持久化向量数据库的目录
    history (str): 要写入的对话内容
    """
    WriteIn(text, history)
    return RAG_Embed(text, persist_directory)


def Retriever(db, question):
    """
    从向量数据库中检索与问题相关的文档
    
    参数:
    db (Chroma): 向量数据库 
    question (str): 查询问题
    
    返回:
    context (str): 检索到的相关文档内容
    """
    # retriever = db.as_retriever()
    # docs = retriever.get_relevant_documents(question)
    docs = db.similarity_search_with_score(question, k=3)
    context = ""
    for doc in docs:
        print(doc[1])
        if doc[1] <= 0.50:
            context = "\n".join(doc[0].page_content)
    return context


def WriteIn(directory, text):
    """
    将对话历史写入文本文件
    
    参数:
    directory (str): 要写入的文本文件路径
    text (str): 要写入的对话内容
    """
    with open(directory, 'w', encoding="utf-8") as f:
        f.write(f"\n" + text)
    f.close()