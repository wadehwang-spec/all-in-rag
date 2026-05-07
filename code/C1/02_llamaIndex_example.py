import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#加载环境变量
load_dotenv()

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

# 接入LLM模型
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/beta",
    is_chat_model=True,
    #数值越低，回答越稳定、保守、一致。
    #数值越高，回答越随机、创造性更强、多样性更高。
    temperature=0.1
)
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

#加载文本数据
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 向量索引
index = VectorStoreIndex.from_documents(docs)

# 查询引擎,在当前这一个检索器（retriever）里，按相似度排序后，返回前 5 个最相关结果
query_engine = index.as_query_engine(similarity_top_k=5)

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?请完整列出"))