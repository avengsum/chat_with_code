import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from .chunk_code import final_chunk

load_dotenv()

Google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


if not Google_api_key:
  raise RuntimeError("Missing Google api key")

if not pinecone_api_key:
  raise RuntimeError("Missing pinecone api key")

embeddings = GoogleGenerativeAIEmbeddings(
  model = "models/text-embedding-004",
  task_type = "RETRIEVAL_DOCUMENT"
)

index_name = "chat-with-code"

print(f"Uploading {len(final_chunk)} chunks to pinecone cloud....")

vector_db = PineconeVectorStore.from_documents(
  documents=final_chunk,
  embedding=embeddings,
  index_name=index_name
)

print("✅ Success! Your code is now stored in the Pinecone cloud.")


