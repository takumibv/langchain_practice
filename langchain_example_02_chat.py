import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo")
chat([
	SystemMessage(content="あなたは優秀なコピーライターです。目を引くキャッチコピーを作成する能力に長けています。"),
	HumanMessage(content="以下の商品説明をもとに、日本向けのLPのトップで使用するためのキャッチコピーを日本語で20文字から35文字程度で5つ作成してください。\n商品説明：スマテスプラスは、デジタルな単語帳です。事前に登録した苦手分野をもとにAIが問題を自動生成して出題してくれるツールです。テスト前に使えば満点間違いなし！"),
])