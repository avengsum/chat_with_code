from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ingestion.embed_store import vector_db

query_embeddings = GoogleGenerativeAIEmbeddings(
  model="models/text-embedding-004",
  task_type="RETRIEVAL_QUERY"
)

base_retriever = vector_db.as_retriever(search_kwargs={"k": 20})

