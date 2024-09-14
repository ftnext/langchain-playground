import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.vectorstores import Chroma

langchain.verbose = True

embedding_function = SentenceTransformerEmbeddings(
    model_name="stsb-xlm-r-multilingual"
)
db = Chroma(
    persist_directory="./imas_chroma_db",
    embedding_function=embedding_function,
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=db.as_retriever()
)

while True:
    sentence = input("入力してください: ")
    if not sentence.strip():
        continue
    result = qa.run(sentence)
    print(result)
    print()
