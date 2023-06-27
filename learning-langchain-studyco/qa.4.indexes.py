import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

langchain.verbose = True

embeddings = OpenAIEmbeddings()
new_db = FAISS.load_local("faiss_index", embeddings)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=new_db.as_retriever()
)

result = qa.run("LangChainの概要を1文で説明してください")
print(result)
