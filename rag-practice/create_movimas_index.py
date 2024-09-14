from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    _split_text_with_regex,
)
from langchain.vectorstores import Chroma
from tqdm import tqdm


class MyCharacterTextSplitter(CharacterTextSplitter):
    def split_text(self, text: str):
        return _split_text_with_regex(
            text, self._separator, self._keep_separator
        )


loader = TextLoader("movie_master.txt")
documents = loader.load()
print(f"{len(documents)=}")

text_splitter = MyCharacterTextSplitter(separator="\n")
docs = text_splitter.split_documents(documents)
print(f"{len(docs)=}")

embedding_function = SentenceTransformerEmbeddings(
    model_name="stsb-xlm-r-multilingual"
)

db = Chroma.from_documents(
    tqdm(docs), embedding_function, persist_directory="./imas_chroma_db"
)
db.persist()
