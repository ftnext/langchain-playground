from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

loader = DirectoryLoader(
    "langchain/docs/_dist/docs_skeleton/docs/get_started/", glob="**/*.md"
)
documents = loader.load()

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(documents, embeddings)

db.save_local("faiss_index")
