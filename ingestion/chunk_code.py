from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from .load_repo import get_repo

try:
   get_repo()
   print("Repo is loaded")
except Exception as e:
   print(f"Error in loading repo {e}")
   raise


get_repo() ## get the repo from git

SUPPORTED_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".go": Language.GO
}

loader = GenericLoader.from_filesystem(
  "./temp_repo",
  glob="**/*",
  suffixes=list(SUPPORTED_EXTENSIONS.keys()),
  parser=LanguageParser()
)

docs = loader.load()

final_chunk = []

for doc in docs:
  lang = doc.metadata.get("language")
  if lang:
    splitter = RecursiveCharacterTextSplitter.from_language(
      language=lang,
      chunk_size=2000,
      chunk_overlap=200
    )
    final_chunk.extend(splitter.split_documents([doc]))

  else:
        print(f"⚠️ Skipping unsupported file: {doc.metadata.get('source')}")



## simple text spiliting used before 
# python_splitter = RecursiveCharacterTextSplitter.from_language(
#   language = Language.PYTHON,
#   chunk_size=1500, 
#   chunk_overlap=150
# )

# texts = python_splitter.split_documents(docs)

