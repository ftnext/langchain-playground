from pprint import pprint

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering.stuff_prompt import system_template
from langchain_chroma import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

embedding_function = HuggingFaceEmbeddings(
    model_name="stsb-xlm-r-multilingual",
    model_kwargs={"revision": "bc1a68705f2e397259207e96349a36ccbc7e6493"},
)
db = Chroma(
    persist_directory="./imas_chroma_db",
    embedding_function=embedding_function,
)

# ref: langchain.chains.question_answering.stuff_prompt.CHAT_PROMPT
# fix KeyError: "Input to ChatPromptTemplate is missing variables {'question'}.  Expected: ['context', 'question'] Received: ['input', 'context']
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{input}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
combine_docs_chain = create_stuff_documents_chain(chat, CHAT_PROMPT)
rag_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)

while True:
    sentence = input("入力してください: ")
    if not sentence.strip():
        continue
    result = rag_chain.invoke({"input": sentence})
    pprint(result, sort_dicts=False)
    print()
    print(result["answer"])
    print()
