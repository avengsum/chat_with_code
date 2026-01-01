from ingestion.embed_store import vector_db
from langchain_community.retrievers import BM25Retriever
from ingestion.chunk_code import final_chunk
from langchain_classic.retrievers import EnsembleRetriever

# query_embeddings = GoogleGenerativeAIEmbeddings(
#   model="models/text-embedding-004",
#   task_type="RETRIEVAL_QUERY"
# )

## implementing hybrid search (keyword + vector)

test_query = "How to handle a GET request in requests?"


bm25_retriever = BM25Retriever.from_documents(final_chunk)

## get 5 snippets
bm25_retriever.k = 20


base_retriever = vector_db.as_retriever(search_kwargs={"k": 20})


hybrid_retriever = EnsembleRetriever(
  retrievers = [bm25_retriever,  base_retriever],
  weights=[0.5,0.5] #Give equal importance
)

actual_data = hybrid_retriever.invoke(test_query)


print(f"here is the hydrid retrvver result: {actual_data} ")

