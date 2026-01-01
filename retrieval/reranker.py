from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from .retriever import hybrid_retriever

compressor = FlashrankRerank(top_n=5)

advance_retriever = ContextualCompressionRetriever(
  base_compressor = compressor,
  base_retriever = hybrid_retriever
)
