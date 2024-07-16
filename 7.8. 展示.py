from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory
import torch
import sys
import time
import RAG
from RealtimeSTT import AudioToTextRecorder
from opencc import OpenCC
import re
from RealtimeTTS import BaseEngine, TextToAudioStream
from gpt_sovits.infer import GPTSoVITSInference
import pyaudio
import logging


class GPTSoVITSEngine(BaseEngine):
    def __init__(
            self,
            model_path=r"D:\Llama3\TTS\gpt_soviet_model",
            bert_path=r"D:\Llama3\TTS\gpt_soviet_model\chinese-roberta-wwm-ext-large",
            cnhubert_path=r"D:\Llama3\TTS\gpt_soviet_model\chinese-hubert-base",
            device=None,
            is_half=True
        ):
        super().__init__()
        # self.can_consume_generators = True
        self.model_path = model_path
        self.inference = GPTSoVITSInference(bert_path, cnhubert_path, device, is_half)
        self.inference.load_sovits(r"D:\Llama3\TTS\gpt_soviet_model\s2G488k.pth")
        self.inference.load_gpt(r"D:\Llama3\TTS\gpt_soviet_model\s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        prompt_text = "飞萤扑火，向死而生。愿我们在清醒的现实再会!"
        self.inference.set_prompt_audio(
            prompt_audio_path=r"D:\Llama3\TTS\sample.wav", 
            prompt_text=prompt_text,
        )
        self.sample_rate = 32000
        self.voice = None
        self.is_half = is_half

    def post_init(self):
        self.engine_name = "GPTSovits"
    def get_stream_info(self):
        return pyaudio.paInt16, 1, self.sample_rate
    
    def _prepare_text_for_synthesis(self, text: str):
        """
        Splits a text into sentences.

        Args:
            text (str): Text to prepare for synthesis.

        Returns:
            text (str): Prepared text.
        """

        logging.debug(f"Preparing text for synthesis: \"{text}\"")

        # A fast fix for last character, may produce weird sounds if it is with text
        text = text.strip()
        text = text.replace("</s>", "")
        text = re.sub("```.*```", "", text, flags=re.DOTALL)
        text = re.sub("`.*`", "", text, flags=re.DOTALL)
        text = re.sub("\(.*\)", "", text, flags=re.DOTALL)
        text = text.replace("```", "")
        text = text.replace("...", " ")
        text = text.replace("»", "")
        text = text.replace("«", "")
        text = re.sub(" +", " ", text)
        # text= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",text)
        # text= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2",text)

        try:
            if len(text) > 2 and text[-1] in ["."]:
                text = text[:-1] 
            elif len(text) > 2 and text[-1] in ["!", "?", ","]:
                text = text[:-1] + " " + text[-1]
            elif len(text) > 3 and text[-2] in ["."]:
                text = text[:-2] 
            elif len(text) > 3 and text[-2] in ["!", "?", ","]:
                text = text[:-2] + " " + text[-2]
        except Exception as e:
            logging.warning(f"Error fixing sentence end punctuation: {e}, Text: \"{text}\"")        

        text = text.strip()

        logging.debug(f"Text after preparation: \"{text}\"")
        return text
    
    def synthesize(self, text: str):
        text = self._prepare_text_for_synthesis(text)
        stream = self.inference.get_tts_wav_stream(text)
        for sample_rate, audio in stream:
            self.queue.put(audio.tobytes())

        return True

    def get_voices(self):
        return [{"name": "zgn"}]

    def set_voice(self, voice):
        if voice:
            self.voice = voice            
            self.inference.load_sovits(f"{self.model_path}/{voice}.pth")
            self.inference.load_gpt(f"{self.model_path}/{voice}.ckpt")
        else:
            raise ValueError(f"Invalid voice: {voice}")

    def set_voice_parameters(self, **voice_parameters):
        if voice_parameters.get("prompt_text") or voice_parameters.get("prompt_audio_path"):
            self.inference.set_prompt_audio(
                prompt_audio_path=voice_parameters.get("prompt_audio_path"),
                prompt_text=voice_parameters.get("prompt_text"),
            )

    def shutdown(self):
        pass

class IntentRecognizer:
    def __init__(self):
        self.keywords = {
            'history': ['根据','他','这些','那些', '这个', '他们'],
            'task':['查询','数据库','表格', '查找', '给我', '我要', '搜索'],
            'clarification': ['没听懂','什么意思','再讲一遍','重复'],
            'goodbye': ['再见','拜拜','退出','下次见']
            
        }

    def recognize_intent(self, user_input):
        user_input = user_input.lower()
        for intent, keywords in self.keywords.items():
            if any(keyword in user_input for keyword in keywords):
                return intent
            
        return 'unknown'

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


examples = [
    {
        "nl_query":"查找所有法国客户的名称和城市",
        "sql_query":"SELECT `customerName`, `city` FROM `customers` WHERE `country` = 'France' LIMIT 5"
    },
    {
        "nl_query":"查找所有信用额度大于10000的客户的名称和信用额度",
        "sql_query":"SELECT `customerName`, `creditLimit` FROM customers WHERE `creditLimit` > 10000 LIMIT 5"
    },
    {
        "nl_query":"一共有多少订单",
        "sql_query":"SELECT COUNT(*) FROM orders"
    },
    {
        "nl_query":"查找所有产品线。",
        "sql_query":"SELECT DISTINCT productLine FROM productlines LIMIT 5"
    },
    {
        "nl_query":"列出所有产品的名称和价格。",
        "sql_query":"SELECT productName, buyPrice FROM products LIMIT 5"
    },
]

#formatted_examples = "\n".join([f"Natural Language Query: {example['nl_query']}\nSQL Query: {example['sql_query']}\n" for example in examples])

def provide_database_info(db):
    # 提取数据库的表结构和列信息
    table_info = db.table_info
         
    # 将表结构和列信息格式化为字符串
    db_info = "Database Tables and Columns:\n"
    db_info += table_info
    
    return db_info

db_info = provide_database_info(db)

print(db_info)

def generate_response(model, user_input, intent, rag):
    resp = None
    context = RAG.Retriever(rag, user_input)
    if intent == 'history':
        resp = talk_with_model_history(model, user_input, context)
    elif intent == 'task':
        resp = talk_with_model_task(model, user_input, context)
    elif intent == 'clarification':
        resp = talk_with_model_clarification(model, user_input, context)
    else:
        resp = talk_with_model_chitchat(model, user_input, context)
    return resp

def talk_with_model_chitchat(model, message, context):
    Messages = [
        {'role': 'system', 'content':f""""我需要你扮演一位全能的AI聊天互动助手的角色。在我们的互动中,你可以在被问到无关数据库查询的问题时进行轻松，愉快的交流并可以适当的开玩笑。以下是我对你的一些期望:

            1. 当我与你进行日常对话时,请用简洁明了的语言回答我的问题。

            2. 除了日常对话,你还应该能够帮助我查询数据库以找到我需要的信息。你不需要真正的去查询数据库信息，你只要告诉我你会这个能力

            3. 在呈现搜索结果时,请以条理清晰、逻辑严谨的方式组织信息。对于复杂的主题,可以适当地提供一些背景知识,以帮助我更好地理解搜索结果的内容。如果搜索结果中包含专业术语或缩略语,请给出相应的解释或定义。

            4. 你的回答都必须以中文呈现。请使用标准的简体中文,避免使用过于口语化或区域性的表达方式。同时,尽量选择简洁明了的词汇和句式,以确保我能够轻松理解你的意思。

            5. 在整个交互过程中,请时刻保持专注、耐心和认真的态度。如果你不确定某个问题的答案,请如实告知,不要试图编造信息或提供不准确的回复。你的诚信和专业性对于我们建立信任关系至关重要。"""},
        
        {'role': 'user', 'content': f"""{message}"""}
    ]
    return talk_with_model(model, Messages)

def talk_with_model_history(model, message, context):
    Messages = [
        {'role': 'system', 'content':f"""You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. You need to consider the following point:
                1.This question is a follow up question with conversation before. You must look up the history and understand what I am asking you. These are the conversation history:{history.messages}
                2.The query results must only contain at most 5 records. Be sure to add the LIMIT 5 clause at the end of the SQL statement to limit the number of results returned.
                3.Never query for all columns from a table. You must query only the columns that are needed to answer the question. 
                4.Pay attention to use only the column names you can see in the tables below. You can never make a query using columns that do not exist. Also, make sure to which column is in which table.
                5.Pay attention to use CURDATE() function get the current date, if the question involves "today".
                6.If you can't find an answer return a query with a polite message.
                7.Ensure the query follows rules:
                - No INSERT, UPDATE, DELETE instructions.
                - No CREATE, ALTER, DROP instructions.
                - Only SELECT queries for data retrieval.

                These are some examples of natural language queries and their corresponding SQL queries retrieved from RAG. You need to consider this information:{context} Use these examples as a reference when generating SQL queries for the user's input question."""},
        
        {'role': 'user', 'content': f"""
                Only use the following tables and columns:{db_info}
                Please refer to the provided examples when generating the SQL query for the following question: {message}
                Use and ONLY use the following exact format to response. DO NOT ADD MORE INFROMATION IN THE RESPONSE: 
                SQL Query: <SQL Query to run>"""}
    ]
    return talk_with_model(model, Messages)

def talk_with_model_clarification(model, message, context):
    Messages = [
        {'role': 'system', 'content': f"""You are playing the role of a MySQL expert. I asked a question and you give a response based on the sql query you generated. Now I want you to follow the following steps to clarify:

            1. Carefully read the part that I pointed out as needing clarification, and ensure that you understand my question.
            2. Provide more details, explanations to make your answer clearer and easier to understand.
            3. After clarifying, ask me if there's anything else that I find unclear or if I need further explanation.
            4. You MUST use Chinese to answer the question.
            5. Do not use any emoji.
         
            These are some examples of natural language queries and their corresponding SQL queries retrieved from RAG. You need to consider this information:{context} Use these examples as a reference when generating SQL queries for the user's input question.
            
            These are the last conversation history:{history.messages}. If the history is null, just tell me you do not find the information that I want you to expalin. Otherwise, clarify all the response AI made based on the steps"""},
        
        {'role': 'user', 'content': f"""{message}"""},
    ]
    return talk_with_model(model, Messages)

def talk_with_model_task(model, message, context):
    # history.clear()
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
                7. !IMPORTANT! When user SPECIFIES the number of results, you should answer with EXACTLY that number of elements you find. If that number is greater than 5, you should only provide at most 5 results and explain clearly.
                
                These are some examples of natural language queries and their corresponding SQL queries retrieved from RAG. You need to consider this information:{context} Use these examples as a reference when generating SQL queries for the user's input question."""},
        
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
    streamer = TextIteratorStreamer(tokenizer)
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

def startRecoder() :
    recorder = AudioToTextRecorder(
        model = "tiny",
        language = "zh",
        spinner = True,
        enable_realtime_transcription = True,
        on_realtime_transcription_update = process_realtime_text,
        wake_words = "jarvis",
        wake_words_sensitivity = 1,
        on_recording_start = on_recording_start,
        on_recording_stop = on_recording_stop,
        on_transcription_start = on_transcription_start,
        wake_word_timeout = 20,
        on_wakeword_detected = on_wakeword_detected,
        on_wakeword_timeout = on_wakeword_timeout,
    )
    return recorder
def on_recording_start():
    print("录音开始...")

def on_recording_stop():
    print("录音结束。")

def on_transcription_start():
    print("开始转录...")

def on_wakeword_detected():
    print("检测到唤醒关键词！")
    speech_response('唤醒')


def on_wakeword_timeout():
    print("唤醒超时，继续等待唤醒词...")

chinese_to_arabic = {
    '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
}

def convert_chinese_to_arabic(text):
    for chinese, arabic in chinese_to_arabic.items():
        text = text.replace(chinese, arabic)
    return text

def process_realtime_text(text):
    converter = OpenCC('t2s')
    real_text = converter.convert(text)
    result = convert_chinese_to_arabic(real_text)
    print(f"实时转录结果: {result}")

def process_final_text(text):
    print(f"最终转录结果是， {text}")
    while stream.is_playing():
        time.sleep(0.1)

def speech_to_text(recorder):
    converter = OpenCC('t2s')
    text = recorder.text()
    final_text = converter.convert(text)
    result = convert_chinese_to_arabic(final_text)
    process_final_text(result)
    recorder.stop()
    return result

def startStream():
    engine = GPTSoVITSEngine()
    stream = TextToAudioStream(engine=engine)
    return stream

def speech_response(text = '介绍', response = ''):
    if text == '介绍':
        stream.feed('您好,我叫jarvis, 请呼唤我的名字来唤醒录音。')
        stream.play_async()
    elif text == '思考':
        stream.feed('正在思考，请作等待。')
        stream.play_async()
    elif text == '错误':
        stream.feed("我没有太理解您的问题，请再次呼唤我的名字，然后重复您的请求。")
        stream.play_async()
    elif text == '唤醒':
        if stream.is_playing():
            stream.stop()
            time.sleep(0.1)
        stream.feed('我在！请问我可以怎么帮您？')
        stream.play_async()
        if stream.is_playing():
            time.sleep(5.25)
    elif text == '结果':
        stream.feed(f"这是我找到的数据，{response}。以上是查询的结果。如有需求，请再次呼唤我的名字。")
        stream.play_async()
    elif text == '再见':
        stream.feed("合作愉快！期待与您下次见面！")
        stream.play_async()
        if stream.is_playing():
            time.sleep(5)
    elif text == '澄清':
        stream.feed(f"请允许我说明，{response}。 如有需求, 请再次呼唤我的名字")
        stream.play_async()
        
prompt = ''
if  __name__ == '__main__':
    model = start_model()
    history = ChatMessageHistory()
    recorder = startRecoder()
    stream = startStream()
    rag = RAG.RAG_Embed('D:\Llama3\RAG\examples.txt', 'db')
    speech_response()
    if stream.is_playing():
        time.sleep(2)
    while True:
        # print(f'我可以怎么帮助你 \n')
        start=time.perf_counter()
        prompt = speech_to_text(recorder)  # 获取用户输入   
        end = time.perf_counter()
        print('STT时间为: %s Seconds'%(end-start))
        temp = IntentRecognizer()
        intent = temp.recognize_intent(prompt)    
        
        if intent == 'goodbye':
            speech_response('再见')
            recorder.shutdown()
            stream.stop()
            break

        speech_response('思考')
        start=time.perf_counter()
        response = generate_response(model, prompt, intent, rag)
        end = time.perf_counter()
        print('第一次运行时间为: %s Seconds'%(end-start))

        if intent == 'unknown' or intent == 'clarification':
            speech_response('澄清', response)
            start=time.perf_counter()
            for char in response:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.05)
            print(f'\n')
            end = time.perf_counter()
        
            print('unknown运行时间为: %s Seconds'%(end-start))
        else:
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
                speech_response('错误')
                continue
                
            print(f'{result} \n')
            history.add_user_message(prompt)

            S2TMessage = [
                {'role': 'system', 'content': f"""You are a MySQL expert. Given an input question, we already find a syntactically correct MySQL query to run, now given you the following user input question,corresponding SQL query, and SQL result, answer the user question with a ntural language response.
                    !IMPORTANT! ONLY show AT MOST 5 groups of data in your response, YOU MAY USE ETC. TO SHOW THERE IS MORE GROUPS OF DATA BESIDES THE ONES THAT YOU HAVE IN YOUR RESPONSE. If the user SPECIFIES the number of results you should find, please PROVIDE EXACTLY THAT NUMBER of results UNLESS it is greater than 5.
                
                    These are the conversation history so far:{history.messages}. Refer to the conversation history if the user has a follow up question.

                    When user SPECIFIES the number of results, you should answer with EXACTLY that number of elements you find. If that number is greater than 5, you should only provide at most 5 results and explain clearly.

                    You MUST use CHINESE to answer the question. When user SPECIFIES the number of results like '1个', '2个', you should answer with EXACTLY that number of elements (one/two results). If that number is greater than 5, you should only provide at most 5 results and explain clearly.
                    """},
                {'role': 'user', 'content': f"""
                    Only use the following result which is the SQL Response: {result}
                    The input question is: {prompt}
                    The SQL Query is: {query}
                    Schema: {db_info}

                    Please consider the conversation history when answering the question.

                    ONLY RESPONSE IN NATURAL LANGUAGE. DO NOT ADD ANY PROMPT LIKE "The answer is" or " The response is". Do not add any sentences like "你所使用的SQL查询是..." in final answer. ONLY provide the final result you find in CHINESE NATURAL LANGUAGE. 
                    """
                }
            ]
            response = talk_with_model(model, S2TMessage)
            speech_response('结果', response) 
            end = time.perf_counter()
            print('程序运行时间为: %s Seconds'%(end-start))
            history.add_ai_message(response)
            print(response)
            for char in response:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.05)
            #print(f'{history.messages}\n')
