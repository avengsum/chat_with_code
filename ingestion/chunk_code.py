from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from ingestion.load_repo import get_repo

docs = get_repo()

python_splitter = RecursiveCharacterTextSplitter.from_language(
  language = Language.PYTHON,
  chunk_size=1500, 
  chunk_overlap=150
)

texts = python_splitter.split_documents(docs)

print(f"Original files: {len(docs)}")
print(f"Resulting chunks: {len(texts)}")
