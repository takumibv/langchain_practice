import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.document_loaders.figma import FigmaFileLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 以下のリンクのFigmaをHTML/CSSに変換するコード
# https://www.figma.com/file/h0sdUKP1SEqiJ8OCwUFxH5/Chat-Dashboard?type=design&node-id=14-884&t=1zbepiR3llG5raOB-4
# 
# 所感
# 例のコードのままだと全くと言っていいほど、精度は高くない
# 考えられる要因
# - figma_doc_retriever.get_relevant_documents で情報が落とされ過ぎている
# - Figmaの作り方


figma_loader = FigmaFileLoader(
    os.environ.get('FIGMA_ACCESS_TOKEN'),
    os.environ.get('FIGMA_NODE_IDS'),
    os.environ.get('FIGMA_FILE_KEY')
)

# see https://python.langchain.com/en/latest/modules/indexes/getting_started.html for more details
index = VectorstoreIndexCreator().from_loaders([figma_loader])
figma_doc_retriever = index.vectorstore.as_retriever()

def generate_code(human_input):
    # I have no idea if the Jon Carmack thing makes for better code. YMMV.
    # See https://python.langchain.com/en/latest/modules/models/chat/getting_started.html for chat info
    system_prompt_template = """You are expert coder Jon Carmack. Use the provided design context to create HTML/CSS code as possible based on the user request.
    Everything must be inline in one file and your response must be directly renderable by the browser.
    Figma file nodes and metadata: {context}"""

    human_prompt_template = "Code the {text}. Ensure it's mobile responsive"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
    # delete the gpt-4 model_name to use the default gpt-3.5 turbo for faster results
    gpt = ChatOpenAI(temperature=.02)
    # Use the retriever's 'get_relevant_documents' method if needed to filter down longer docs
    relevant_nodes = figma_doc_retriever.get_relevant_documents(human_input)

    conversation = [system_message_prompt, human_message_prompt]
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    chat_prompt = ChatPromptTemplate.from_messages(conversation)

    print(chat_prompt.format_prompt( 
        context=relevant_nodes, 
        text=human_input).to_messages())

    response = gpt(chat_prompt.format_prompt( 
        context=relevant_nodes, 
        text=human_input).to_messages())
    return response

response = generate_code("all page")


print("==========")
print(response.content)

with open("index.html", mode='w') as f:
    f.write(response.content)
