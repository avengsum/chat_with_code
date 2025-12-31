from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from .load_repo import get_repo

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


print("Total chunks:", len(final_chunk))
first_chunks = final_chunk[0]
print("Chunk content:")
print(first_chunks.page_content)
print("Source file:", first_chunks.metadata.get("source"))
print("Language:", first_chunks.metadata.get("language"))


## simple text spiliting used before 
# python_splitter = RecursiveCharacterTextSplitter.from_language(
#   language = Language.PYTHON,
#   chunk_size=1500, 
#   chunk_overlap=150
# )

# texts = python_splitter.split_documents(docs)

