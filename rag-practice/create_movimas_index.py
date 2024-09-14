from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters.character import _split_text_with_regex
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

embedding_function = HuggingFaceEmbeddings(
    model_name="stsb-xlm-r-multilingual"
)

db = Chroma.from_documents(
    tqdm(docs), embedding_function, persist_directory="./imas_chroma_db"
)
