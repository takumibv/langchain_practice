# JSONファイルを読み込むサンプル

import os

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.document_loaders import JSONLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import json
from pathlib import Path
from pprint import pprint

file_path='./figma_chat_short.json'
json_raw_text = Path(file_path).read_text()
data = json.loads(json_raw_text)

text_file_path = "./_data.txt"
with open(text_file_path, "w") as f:
    f.write(json_raw_text)
    f.close()

loader = TextLoader(text_file_path)

# loader = JSONLoader(
#     file_path=file_path,
#     jq_schema=".name",
#     )
# data = loader.load()

# pprint(data)

text_splitter = CharacterTextSplitter(
    separator = ",\n",
    chunk_size = 400,
    chunk_overlap = 0,
    length_function = len,
)
# text_list = text_splitter.split_text("json data is:\n" + json_raw_text)

# print(text_list)
# print(len(text_list))

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

query = "JSONにの内容をHTML/CSSに変換してください"
# query = "Figmaに表示されているユーザの名前は何ですか？"
print(f"\n\n{query}")
answer = index.query(query)
print("====== index.query ======")
pprint(answer)

# 関連文章の検索
retriever = index.vectorstore.as_retriever()
docs = retriever.get_relevant_documents(query)
print("====== retriever.get_relevant_documents ======")
pprint(docs[0].page_content)
