from modelscope import snapshot_download

cache_dir = r'C:\Users\bugbank\Desktop\Llama3'

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct',cache_dir=cache_dir)