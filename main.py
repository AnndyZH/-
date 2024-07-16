from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain_community.utilities.sql_database import SQLDatabase
import torch
import sys
import time
import RAG
from threading import Thread
import shutil
#import re



# 数据库连接信息
db_user = "root"
db_password = "Moule123!"
db_host = 'localhost'
db_name = "classicmodels"

# 创建数据库连接对象
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# 打印数据库中的表信息

# 模型目录和设备配置
model_dir = r'D:\Llama3\LLM-Research\Meta-Llama-3-8B-Instruct'
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# 加载分词器和模型

def start_model():
    """
    从本地加载语言模型
    
    返回:
    model: 加载好的语言模型
    """
    print("正在从本地加载模型...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map=device)
    print("完成本地模型的加载")
    return model

def provide_database_info(db):
    """
    提取数据库的表结构和列信息
    
    参数:
    db (SQLDatabase): 数据库连接对象
    
    返回:
    db_info (str): 格式化后的数据库表结构和列信息
    """
    table_info = db.table_info
    db_info = "Database Tables and Columns:\n"
    db_info += table_info
    
    return db_info

db_info = provide_database_info(db)

def generate_response(model, user_input, ragcon, raghis, result, num):
    """
    生成模型响应
    
    参数:
    model: 语言模型
    user_input (str): 用户输入
    ragcon (Chroma): 知识库向量数据库
    raghis (Chroma): 历史对话向量数据库
    result: SQL查询结果
    num (int): 响应生成模式,1为生成SQL查询,2为生成自然语言回复
    
    返回:
    response (str): 模型生成的响应
    """
    context = RAG.Retriever(ragcon, user_input)
    chat = RAG.Retriever(raghis, user_input)
    if num == 1:
        return talk_with_model_T2S(model, user_input, context, chat)
    else:
        return talk_with_model_S2T(model, result, user_input, chat)

def talk_with_model_S2T(model, result, message, chat):
    """
    根据SQL查询结果生成自然语言回复
    
    参数:
    model: 语言模型
    result: SQL查询结果
    message (str): 用户输入问题
    chat (str): 历史对话信息
    
    返回:
    response (str): 模型生成的自然语言回复
    """
    S2TMessage = [
            {'role': 'system', 'content': f"""You are a MySQL expert. Given an input question, we already find a syntactically correct MySQL query to run, now given you the following user input question,corresponding SQL query, and SQL result, answer the user question with a ntural language response.
                !IMPORTANT! MUST show AT MOST 5 groups of data in your response. 
            
                These are the conversation history so far:{chat}. 
                You MUST use Chinese to answer the question!"""},
            {'role': 'user', 'content': f"""
                This is the SQL response {result}.Only use the following response to generate answer.
                The input question is: {message}
                You need to answer in natural language. Do not explain the process you did, you only need to give me the answer.
                    
                """}
            ]
    return talk_with_model(model, S2TMessage)

def talk_with_model_T2S(model, message, context, chat):
    """
    根据用户输入生成SQL查询语句
    
    参数:
    model: 语言模型
    message (str): 用户输入问题
    context (str): 从知识库检索到的相关信息
    chat (str): 历史对话信息
    
    返回:
    response (str): 模型生成的SQL查询语句
    """
    print(chat)
    Messages = [
            {'role': 'system', 'content': f"""You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. You need to consider the following point:
                1.The query results must only contain at most 5 records. Be sure to add the LIMIT 5 clause at the end of the SQL statement to limit the number of results returned.
                2.Never query for all columns from a table. You must query only the columns that are needed to answer the question. 
                3.Pay attention to use only the column names you can see in the tables below. You can never make a query using columns that do not exist. Also, make sure to which column is in which table.
                4.Pay attention to use CURDATE() function get the current date, if the question involves "today".
                5.If you can't find an answer return a query with a polite message.
                6.Ensure the query follows rules:
                - No INSERT, UPDATE, DELETE instructions.
                - No CREATE, ALTER, DROP instructions.
                - Only SELECT queries for data retrieval.
                7.Only answer in CHINESE. This is important!
                These are some examples of natural language queries and their corresponding SQL queries retrieved from RAG. You need to consider this information:{context} Use these examples as a reference when generating SQL queries for the user's input question.
                These are the history message of this conversation, you should consider the history chat:{chat}"""},
                
            {'role': 'user', 'content': f"""
                Only use the following tables and columns:{db_info}
                Please refer to the provided examples when generating the SQL query for the following question: {message}
                Use and ONLY use the following exact format to response. DO NOT ADD MORE INFROMATION IN THE RESPONSE: 
                SQL Query: <SQL Query to run>
                """},
    ]
    return talk_with_model(model, Messages)

def talk_with_model(model, message):
    """
    与语言模型进行对话
    
    参数:
    model: 语言模型
    message (list): 对话消息列表
    
    返回:
    response (str): 模型生成的响应
    """
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 将文本转为模型输入
    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)

    # 生成模型输出
    generated_ids = model.generate(
        model_input.input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        top_k = 50,
        top_p = 0.95,
        num_beams = 1,
        temperature=0.6,
    )

    # 处理生成的ID，去掉输入部分，保留生成部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
    
    # 解码生成的文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def Clean_History(directory):
    """
    清空历史对话文件
    
    参数:
    directory (str): 历史对话文件路径
    """
    with open(directory, 'a', encoding='utf-8') as hist:
        hist.truncate(0)

if  __name__ == '__main__':
    shutil.rmtree("db2")
    model = start_model()
    historymessage = ""
    Clean_History('D:\Llama3\RAG\history.txt')
    knowledgebase_rag = RAG.RAG_Embed_Knowledgebase('D:\Llama3\RAG\examples.txt', 'db')
    history_rag = RAG.RAG_Embed_History('D:\Llama3\RAG\history.txt', 'db2', "history")
    while True:

        print(f'我可以怎么帮助你 \n')
        message = input()  # 获取用户输入   
        historymessage += message +f"\n"
        if message == "再见":
            break

        start = time.perf_counter()

        response = generate_response(model, message, knowledgebase_rag, history_rag, None, 1)

        end = time.perf_counter()
        print(f'第一次运行时间为: %s Seconds \n'%(end-start))
        
        
        query = response[len("SQL Query: "):]
        query = query.strip(";")
        
        # query = query + " LIMIT 5"
        print(f'{query} \n')

        #print(f'{response} \n')
        try:
            start=time.perf_counter()
            result = db.run(query)
            end = time.perf_counter()
            print(f'查询数据库运行时间为: %s Seconds \n'%(end-start))
        except Exception:
            print("我没有太理解你的问题，你可以再说一遍吗")
            continue
                
            
        print(f'{result} \n')
            
        
        start=time.perf_counter()
        response = generate_response(model, message, knowledgebase_rag, history_rag, result, 2) 
        end = time.perf_counter()
        historymessage += response +f"\n"
        print('程序运行时间为: %s Seconds'%(end-start))
        
        RAG.RAG_Embed_History('D:\Llama3\RAG\history.txt', 'db2', historymessage)
