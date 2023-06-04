import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
query_result = embeddings.embed_query("ITエンジニアについて30文字で教えて。")

print(query_result)
print(len(query_result))