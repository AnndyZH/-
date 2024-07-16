from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, TextStreamer
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory
import torch
import sys
import time
import RAG
from threading import Thread
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
    print("正在从本地加载模型...")

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map=device)
    print("完成本地模型的加载")
    return model

def provide_database_info(db):
    # 提取数据库的表结构和列信息
    table_info = db.table_info
         
    # 将表结构和列信息格式化为字符串
    db_info = "Database Tables and Columns:\n"
    db_info += table_info
    
    return db_info

db_info = provide_database_info(db)

print(db_info)

def generate_response(model, user_input, ragcon, raghis):
    context = RAG.Retriever(ragcon, user_input)
    chat = RAG.Retriever(raghis, user_input)
    resp = talk_with_model_task(model, user_input, context, chat)

    return resp



def talk_with_model_task(model, message, context, chat):
    print(context)
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
        max_new_tokens=512,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        top_k = 50,
        top_p = 0.95,
        num_beams = 1,
        temperature=0.5,
    )

    # 处理生成的ID，去掉输入部分，保留生成部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
    
    # 解码生成的文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



if  __name__ == '__main__':
    model = start_model()
    history = ChatMessageHistory()
    example_rag = RAG.RAG_Embed('D:\Llama3\RAG\examples.txt', 'db')
    history_rag = RAG.RAG_Embed('D:\Llama3\RAG\history.txt', 'db2')
    while True:

        print(f'我可以怎么帮助你 \n')
        message = input()  # 获取用户输入   
        
        if message == "再见":
            break

        start=time.perf_counter()

        response = generate_response(model, message, example_rag, history_rag)

        end = time.perf_counter()
        print('第一次运行时间为: %s Seconds'%(end-start))
        
        start=time.perf_counter()
        query = response[len("SQL Query: "):]
        query = query.strip(";")
        # query = query + " LIMIT 5"
        print(f'{query} \n')

        # 输出生成的响应
        #print(f'{response} \n')
        try:
            result = db.run(query)
        except Exception:
            print("我没有太理解你的问题，你可以再说一遍吗")
            continue
                
            
        print(f'{result} \n')
        history.add_user_message(message)
            

        S2TMessage = [
            {'role': 'system', 'content': f"""You are a MySQL expert. Given an input question, we already find a syntactically correct MySQL query to run, now given you the following user input question,corresponding SQL query, and SQL result, answer the user question with a ntural language response.
                !IMPORTANT! MUST show AT MOST 5 groups of data in your response, YOU MAY USE ETC. TO SHOW THERE IS MORE GROUPS OF DATA BESIDES THE ONES THAT YOU HAVE IN YOUR RESPONSE.
            
                These are the conversation history so far:{history.messages}. Refer to the conversation history if the user has a follow up question.
                You MUST use Chinese to answer the question."""},
            {'role': 'user', 'content': f"""
                Only use the following result which is the SQL Response: {result}
                The input question is: {message}
                The SQL Query is: {query}
                    
                Please consider the conversation history when answering the question.

                ONLY RESPONSE IN NATURAL LANGUAGE. DO NOT ADD ANY PROMPT LIKE "The answer is" or " The response is"."""}
            ]

        response = talk_with_model(model, S2TMessage) 
        end = time.perf_counter()
        print('程序运行时间为: %s Seconds'%(end-start))
        history.add_ai_message(response)
        #RAG.WriteIn("history.txt", history.messages)
        history.clear()
        # for char in response:
        #     sys.stdout.write(char)
        #     sys.stdout.flush()
        #     time.sleep(0.05)
            #print(f'{history.messages}\n')
