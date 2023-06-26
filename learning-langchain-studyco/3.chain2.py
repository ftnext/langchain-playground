import langchain
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

langchain.verbose = True

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

cot_template = """\
以下の質問に回答してください。

### 質問 ###
{question}
### 質問終了 ###

ステップバイステップで考えましょう。
"""
cot_human_message = HumanMessagePromptTemplate.from_template(cot_template)
cot_prompt = ChatPromptTemplate.from_messages([cot_human_message])
cot_chain = LLMChain(llm=chat, prompt=cot_prompt)

summarize_template = """\
入力を結論だけ一言に要約してください。

### 入力 ###
{input}
### 入力終了 ###
"""
summarize_human_message = HumanMessagePromptTemplate.from_template(
    summarize_template
)
summarize_prompt = ChatPromptTemplate.from_messages([summarize_human_message])
summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

cot_summarize_chain = SimpleSequentialChain(
    chains=[cot_chain, summarize_chain]
)

result = cot_summarize_chain(
    "私は市場に行って10個のリンゴを買いました。隣人に2つ、修理工に2つ渡しました。それから5つのリンゴを買って1つ食べました。残りは何個ですか？"
)
print(result["output"])
