import os

# LangChain Prompt の機能のチュートリアル

from langchain import PromptTemplate
from langchain.llms import OpenAI

template = """
{subject}について{number}文字で教えて。

"""

prompt = PromptTemplate(
    input_variables=["subject", "number"],
    template=template,
)
prompt_text = prompt.format(subject="ITエンジニア", number="30")
print(prompt_text)

llm = OpenAI(model_name="text-davinci-003")
print(llm(prompt_text))