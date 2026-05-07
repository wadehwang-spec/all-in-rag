import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode
#from llama_index.core.query_engine.pandas import PandasQueryEngine
from llama_index.experimental.query_engine import PandasQueryEngine
#from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

# 配置模型
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1.加载数据并为每个工作表创建查询引擎和摘要节点
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)
    
    # 为当前工作表创建一个摘要节点（IndexNode）
    year = sheet_name.replace('年份_', '')
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine

# 2. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)

# 3. 创建递归检索器
# 3.1 创建顶层检索器，用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 3.2 创建递归检索器
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True,
)

# 4. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 5. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")
response = query_engine.query(query)
print(f"回答: {response}")







#备注当前环境下的组件包信息


# (all-in-rag) ubuntu@ca6bd8c590c4:/workspace/code/C3$ conda list
# # packages in environment at /home/ubuntu/miniconda3/envs/all-in-rag:
# #
# # Name                                      Version          Build            Channel
# _libgcc_mutex                               0.1              main
# _openmp_mutex                               5.1              1_gnu
# accelerate                                  1.13.0           pypi_0           pypi
# aiofiles                                    25.1.0           pypi_0           pypi
# aiohappyeyeballs                            2.6.1            pypi_0           pypi
# aiohttp                                     3.13.5           pypi_0           pypi
# aiosignal                                   1.4.0            pypi_0           pypi
# aiosqlite                                   0.22.1           pypi_0           pypi
# annotated-doc                               0.0.4            pypi_0           pypi
# annotated-types                             0.7.0            pypi_0           pypi
# antlr4-python3-runtime                      4.9.3            pypi_0           pypi
# anyio                                       4.13.0           pypi_0           pypi
# apscheduler                                 3.11.2           pypi_0           pypi
# attrs                                       26.1.0           pypi_0           pypi
# azure-core                                  1.39.0           pypi_0           pypi
# azure-identity                              1.25.3           pypi_0           pypi
# backoff                                     2.2.1            pypi_0           pypi
# banks                                       2.4.2            pypi_0           pypi
# bcrypt                                      5.0.0            pypi_0           pypi
# beautifulsoup4                              4.13.5           pypi_0           pypi
# bilibili-api-python                         17.3.0           pypi_0           pypi
# brotli                                      1.1.0            pypi_0           pypi
# build                                       1.4.4            pypi_0           pypi
# bzip2                                       1.0.8            h5eee18b_6
# ca-certificates                             2026.3.19        h06a4308_0
# certifi                                     2026.4.22        pypi_0           pypi
# cffi                                        2.0.0            pypi_0           pypi
# charset-normalizer                          3.4.7            pypi_0           pypi
# chromadb                                    1.5.8            pypi_0           pypi
# click                                       8.3.3            pypi_0           pypi
# cohere                                      5.21.1           pypi_0           pypi
# colorama                                    0.4.6            pypi_0           pypi
# contourpy                                   1.3.3            pypi_0           pypi
# cryptography                                47.0.0           pypi_0           pypi
# cycler                                      0.12.1           pypi_0           pypi
# dataclasses-json                            0.6.7            pypi_0           pypi
# datasets                                    4.0.0            pypi_0           pypi
# defusedxml                                  0.7.1            pypi_0           pypi
# deprecated                                  1.3.1            pypi_0           pypi
# dill                                        0.3.8            pypi_0           pypi
# dirtyjson                                   1.0.8            pypi_0           pypi
# distro                                      1.9.0            pypi_0           pypi
# duckdb                                      1.5.2            pypi_0           pypi
# durationpy                                  0.10             pypi_0           pypi
# effdet                                      0.4.1            pypi_0           pypi
# einops                                      0.8.1            pypi_0           pypi
# emoji                                       2.15.0           pypi_0           pypi
# et-xmlfile                                  2.0.0            pypi_0           pypi
# eval-type-backport                          0.3.1            pypi_0           pypi
# expat                                       2.7.5            h7354ed3_0
# faiss-cpu                                   1.13.2           pypi_0           pypi
# fastavro                                    1.12.2           pypi_0           pypi
# filelock                                    3.29.0           pypi_0           pypi
# filetype                                    1.2.0            pypi_0           pypi
# flatbuffers                                 25.12.19         pypi_0           pypi
# fonttools                                   4.62.1           pypi_0           pypi
# frozenlist                                  1.8.0            pypi_0           pypi
# fsspec                                      2025.3.0         pypi_0           pypi
# ftfy                                        6.3.1            pypi_0           pypi
# google-api-core                             2.30.3           pypi_0           pypi
# google-auth                                 2.49.2           pypi_0           pypi
# google-cloud-vision                         3.13.0           pypi_0           pypi
# googleapis-common-protos                    1.74.0           pypi_0           pypi
# greenlet                                    3.5.0            pypi_0           pypi
# griffe                                      2.0.2            pypi_0           pypi
# griffecli                                   2.0.2            pypi_0           pypi
# griffelib                                   2.0.2            pypi_0           pypi
# grpcio                                      1.67.1           pypi_0           pypi
# grpcio-status                               1.67.1           pypi_0           pypi
# h11                                         0.16.0           pypi_0           pypi
# hf-xet                                      1.4.3            pypi_0           pypi
# html5lib                                    1.1              pypi_0           pypi
# httpcore                                    1.0.9            pypi_0           pypi
# httptools                                   0.7.1            pypi_0           pypi
# httpx                                       0.25.2           pypi_0           pypi
# httpx-sse                                   0.4.3            pypi_0           pypi
# huggingface-hub                             0.36.2           pypi_0           pypi
# idna                                        3.13             pypi_0           pypi
# importlib-metadata                          8.7.1            pypi_0           pypi
# importlib-resources                         7.1.0            pypi_0           pypi
# jinja2                                      3.1.6            pypi_0           pypi
# jiter                                       0.14.0           pypi_0           pypi
# joblib                                      1.5.3            pypi_0           pypi
# jsonpatch                                   1.33             pypi_0           pypi
# jsonpath-python                             1.1.5            pypi_0           pypi
# jsonpointer                                 3.1.1            pypi_0           pypi
# jsonschema                                  4.26.0           pypi_0           pypi
# jsonschema-specifications                   2025.9.1         pypi_0           pypi
# kiwisolver                                  1.5.0            pypi_0           pypi
# kubernetes                                  35.0.0           pypi_0           pypi
# langchain                                   0.3.26           pypi_0           pypi
# langchain-community                         0.3.27           pypi_0           pypi
# langchain-core                              0.3.71           pypi_0           pypi
# langchain-deepseek                          0.1.4            pypi_0           pypi
# langchain-experimental                      0.3.4            pypi_0           pypi
# langchain-huggingface                       0.3.1            pypi_0           pypi
# langchain-openai                            0.3.28           pypi_0           pypi
# langchain-text-splitters                    0.3.8            pypi_0           pypi
# langdetect                                  1.0.9            pypi_0           pypi
# langsmith                                   0.7.38           pypi_0           pypi
# lark                                        1.2.2            pypi_0           pypi
# lazy-loader                                 0.4              pypi_0           pypi
# ld_impl_linux-64                            2.44             h9e0c5a2_3
# libexpat                                    2.7.5            h7354ed3_0
# libffi                                      3.4.8            hc5d346e_2
# libgcc                                      15.2.0           h69a1729_7
# libgcc-ng                                   15.2.0           h166f726_7
# libgomp                                     15.2.0           h4751f2c_7
# libstdcxx                                   15.2.0           h39759b7_7
# libuuid                                     1.41.5           h5eee18b_0
# libxcb                                      1.17.0           h9b100fa_0
# libzlib                                     1.3.1            h47b2149_1
# llama-cloud                                 1.6.0            pypi_0           pypi
# llama-cloud-services                        0.6.54           pypi_0           pypi
# llama-index-agent-openai                    0.4.12           pypi_0           pypi
# llama-index-cli                             0.5.5            pypi_0           pypi
# llama-index-core                            0.10.55          pypi_0           pypi
# llama-index-embeddings-adapter              0.4.1            pypi_0           pypi
# llama-index-embeddings-huggingface          0.2.1            pypi_0           pypi
# llama-index-embeddings-openai               0.1.11           pypi_0           pypi
# llama-index-experimental                    0.2.0            pypi_0           pypi
# llama-index-indices-managed-llama-cloud     0.2.7            pypi_0           pypi
# llama-index-instrumentation                 0.5.0            pypi_0           pypi
# llama-index-legacy                          0.9.48.post4     pypi_0           pypi
# llama-index-llms-azure-openai               0.4.2            pypi_0           pypi
# llama-index-llms-deepseek                   0.1.2            pypi_0           pypi
# llama-index-llms-openai                     0.1.26           pypi_0           pypi
# llama-index-llms-openai-like                0.1.3            pypi_0           pypi
# llama-index-multi-modal-llms-openai         0.1.9            pypi_0           pypi
# llama-index-postprocessor-cohere-rerank     0.5.1            pypi_0           pypi
# llama-index-program-openai                  0.3.2            pypi_0           pypi
# llama-index-question-gen-openai             0.3.1            pypi_0           pypi
# llama-index-readers-file                    0.1.33           pypi_0           pypi
# llama-index-readers-llama-parse             0.1.6            pypi_0           pypi
# llama-index-workflows                       2.20.0           pypi_0           pypi
# llama-parse                                 0.4.9            pypi_0           pypi
# lxml                                        5.4.0            pypi_0           pypi
# markdown                                    3.8.2            pypi_0           pypi
# markdown-it-py                              4.0.0            pypi_0           pypi
# markupsafe                                  3.0.3            pypi_0           pypi
# marshmallow                                 3.26.2           pypi_0           pypi
# matplotlib                                  3.10.9           pypi_0           pypi
# mdurl                                       0.1.2            pypi_0           pypi
# milvus-lite                                 2.5.1            pypi_0           pypi
# mistralai                                   0.1.8            pypi_0           pypi
# ml-dtypes                                   0.5.4            pypi_0           pypi
# mmh3                                        5.2.1            pypi_0           pypi
# mpmath                                      1.3.0            pypi_0           pypi
# msal                                        1.36.0           pypi_0           pypi
# msal-extensions                             1.3.1            pypi_0           pypi
# multidict                                   6.7.1            pypi_0           pypi
# multiprocess                                0.70.16          pypi_0           pypi
# mypy-extensions                             1.1.0            pypi_0           pypi
# ncurses                                     6.5              h7934f7d_0
# nest-asyncio                                1.6.0            pypi_0           pypi
# networkx                                    3.6.1            pypi_0           pypi
# nltk                                        3.9.4            pypi_0           pypi
# numpy                                       1.26.4           pypi_0           pypi
# nvidia-cublas-cu12                          12.4.5.8         pypi_0           pypi
# nvidia-cuda-cupti-cu12                      12.4.127         pypi_0           pypi
# nvidia-cuda-nvrtc-cu12                      12.4.127         pypi_0           pypi
# nvidia-cuda-runtime-cu12                    12.4.127         pypi_0           pypi
# nvidia-cudnn-cu12                           9.1.0.70         pypi_0           pypi
# nvidia-cufft-cu12                           11.2.1.3         pypi_0           pypi
# nvidia-curand-cu12                          10.3.5.147       pypi_0           pypi
# nvidia-cusolver-cu12                        11.6.1.9         pypi_0           pypi
# nvidia-cusparse-cu12                        12.3.1.170       pypi_0           pypi
# nvidia-cusparselt-cu12                      0.6.2            pypi_0           pypi
# nvidia-nccl-cu12                            2.21.5           pypi_0           pypi
# nvidia-nvjitlink-cu12                       12.4.127         pypi_0           pypi
# nvidia-nvtx-cu12                            12.4.127         pypi_0           pypi
# oauthlib                                    3.3.1            pypi_0           pypi
# olefile                                     0.47             pypi_0           pypi
# omegaconf                                   2.3.0            pypi_0           pypi
# onnx                                        1.21.0           pypi_0           pypi
# onnxruntime                                 1.25.1           pypi_0           pypi
# openai                                      1.109.1          pypi_0           pypi
# opencv-python                               4.13.0.92        pypi_0           pypi
# opencv-python-headless                      4.12.0.88        pypi_0           pypi
# openpyxl                                    3.1.5            pypi_0           pypi
# openssl                                     3.5.6            h1b28b03_0
# opentelemetry-api                           1.39.1           pypi_0           pypi
# opentelemetry-exporter-otlp-proto-common    1.39.1           pypi_0           pypi
# opentelemetry-exporter-otlp-proto-grpc      1.39.1           pypi_0           pypi
# opentelemetry-proto                         1.39.1           pypi_0           pypi
# opentelemetry-sdk                           1.39.1           pypi_0           pypi
# opentelemetry-semantic-conventions          0.60b1           pypi_0           pypi
# orjson                                      3.10.18          pypi_0           pypi
# overrides                                   7.7.0            pypi_0           pypi
# packaging                                   26.0             py312h06a4308_0
# pandas                                      2.2.3            pypi_0           pypi
# pdf2image                                   1.17.0           pypi_0           pypi
# pdfminer-six                                20260107         pypi_0           pypi
# pi-heif                                     1.3.0            pypi_0           pypi
# pikepdf                                     10.5.1           pypi_0           pypi
# pillow                                      11.2.1           pypi_0           pypi
# pip                                         26.0.1           pyhc872135_1
# platformdirs                                4.9.6            pypi_0           pypi
# polars                                      1.40.1           pypi_0           pypi
# polars-runtime-32                           1.40.1           pypi_0           pypi
# propcache                                   0.4.1            pypi_0           pypi
# proto-plus                                  1.27.2           pypi_0           pypi
# protobuf                                    5.29.6           pypi_0           pypi
# psutil                                      7.2.2            pypi_0           pypi
# pthread-stubs                               0.3              h0ce48e5_1
# pyarrow                                     20.0.0           pypi_0           pypi
# pyasn1                                      0.6.3            pypi_0           pypi
# pyasn1-modules                              0.4.2            pypi_0           pypi
# pybase64                                    1.4.3            pypi_0           pypi
# pycocotools                                 2.0.11           pypi_0           pypi
# pycparser                                   3.0              pypi_0           pypi
# pycryptodomex                               3.23.0           pypi_0           pypi
# pydantic                                    2.8.2            pypi_0           pypi
# pydantic-core                               2.20.1           pypi_0           pypi
# pydantic-settings                           2.14.0           pypi_0           pypi
# pygments                                    2.20.0           pypi_0           pypi
# pyjwt                                       2.10.1           pypi_0           pypi
# pymilvus                                    2.5.11           pypi_0           pypi
# pymilvus-model                              0.3.2            pypi_0           pypi
# pyparsing                                   3.3.2            pypi_0           pypi
# pypdf                                       4.3.1            pypi_0           pypi
# pypdfium2                                   5.7.1            pypi_0           pypi
# pypika                                      0.51.1           pypi_0           pypi
# pyproject-hooks                             1.2.0            pypi_0           pypi
# python                                      3.12.7           h5148396_0
# python-dateutil                             2.9.0.post0      pypi_0           pypi
# python-dotenv                               1.2.2            pypi_0           pypi
# python-iso639                               2026.4.20        pypi_0           pypi
# python-magic                                0.4.27           pypi_0           pypi
# python-multipart                            0.0.27           pypi_0           pypi
# python-oxmsg                                0.0.2            pypi_0           pypi
# pytz                                        2026.1.post1     pypi_0           pypi
# pyyaml                                      6.0.3            pypi_0           pypi
# qrcode                                      8.2              pypi_0           pypi
# qrcode-terminal                             0.8              pypi_0           pypi
# rapidfuzz                                   3.14.5           pypi_0           pypi
# readline                                    8.3              hc2a1206_0
# referencing                                 0.37.0           pypi_0           pypi
# regex                                       2026.4.4         pypi_0           pypi
# requests                                    2.33.1           pypi_0           pypi
# requests-oauthlib                           2.0.0            pypi_0           pypi
# requests-toolbelt                           1.0.0            pypi_0           pypi
# rich                                        15.0.0           pypi_0           pypi
# rpds-py                                     0.30.0           pypi_0           pypi
# safetensors                                 0.7.0            pypi_0           pypi
# scikit-learn                                1.8.0            pypi_0           pypi
# scipy                                       1.17.1           pypi_0           pypi
# sentence-transformers                       2.7.0            pypi_0           pypi
# setuptools                                  82.0.1           py312h06a4308_0
# shellingham                                 1.5.4            pypi_0           pypi
# six                                         1.17.0           pypi_0           pypi
# sniffio                                     1.3.1            pypi_0           pypi
# soupsieve                                   2.8.3            pypi_0           pypi
# sqlalchemy                                  2.0.49           pypi_0           pypi
# sqlite                                      3.51.2           h3e8d24a_0
# striprtf                                    0.0.26           pypi_0           pypi
# sympy                                       1.13.1           pypi_0           pypi
# tenacity                                    8.5.0            pypi_0           pypi
# threadpoolctl                               3.6.0            pypi_0           pypi
# tiktoken                                    0.12.0           pypi_0           pypi
# timm                                        1.0.26           pypi_0           pypi
# tinytag                                     2.2.1            pypi_0           pypi
# tk                                          8.6.15           h54e0aa7_0
# tokenizers                                  0.22.2           pypi_0           pypi
# torch                                       2.6.0            pypi_0           pypi
# torchaudio                                  2.6.0            pypi_0           pypi
# torchvision                                 0.21.0           pypi_0           pypi
# tqdm                                        4.67.3           pypi_0           pypi
# transformers                                4.57.6           pypi_0           pypi
# triton                                      3.2.0            pypi_0           pypi
# typer                                       0.25.0           pypi_0           pypi
# types-requests                              2.33.0.20260408  pypi_0           pypi
# typing-extensions                           4.15.0           pypi_0           pypi
# typing-inspect                              0.9.0            pypi_0           pypi
# typing-inspection                           0.4.2            pypi_0           pypi
# tzdata                                      2026.2           pypi_0           pypi
# tzlocal                                     5.3.1            pypi_0           pypi
# ujson                                       5.12.0           pypi_0           pypi
# unstructured                                0.18.11          pypi_0           pypi
# unstructured-client                         0.41.0           pypi_0           pypi
# unstructured-inference                      1.0.5            pypi_0           pypi
# unstructured-pytesseract                    0.3.15           pypi_0           pypi
# urllib3                                     2.6.3            pypi_0           pypi
# uuid-utils                                  0.14.1           pypi_0           pypi
# uvicorn                                     0.46.0           pypi_0           pypi
# uvloop                                      0.22.1           pypi_0           pypi
# visual-bge                                  0.1.0            pypi_0           pypi
# watchfiles                                  1.1.1            pypi_0           pypi
# wcwidth                                     0.6.0            pypi_0           pypi
# webencodings                                0.5.1            pypi_0           pypi
# websocket-client                            1.9.0            pypi_0           pypi
# websockets                                  16.0             pypi_0           pypi
# wheel                                       0.46.3           py312h06a4308_0
# wrapt                                       2.1.2            pypi_0           pypi
# xorg-libx11                                 1.8.12           h9b100fa_1
# xorg-libxau                                 1.0.12           h9b100fa_0
# xorg-libxdmcp                               1.1.5            h9b100fa_0
# xorg-xorgproto                              2024.1           h5eee18b_1
# xxhash                                      3.7.0            pypi_0           pypi
# xz                                          5.8.2            h448239c_0
# yarl                                        1.20.1           pypi_0           pypi
# zipp                                        3.23.1           pypi_0           pypi
# zlib                                        1.3.1            h47b2149_1
# zstandard                                   0.25.0           pypi_0           pypi