from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from retrieval.reranker import advance_retriever

load_dotenv()

Google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  temperature = 0.1,
  max_retries= 2
)

template = """
You are a Senior Engineer. Answer using ONLY the following code snippets. 
If you can't find the answer, say "Logic not found."

CONTEXT:
{context}

QUESTION:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """
    Takes the raw code snippets and turns them into 
    one clean string for the AI to read.
    """
    formatted_list = []
    for d in docs:
        source_file = d.metadata.get('source')
        code_content = d.page_content
        block = f"--- SOURCE FILE: {source_file} ---\n{code_content}"
        formatted_list.append(block)
    
    return "\n\n".join(formatted_list)

rag_chain = (
  {"context": advance_retriever | format_docs, "question" :RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

question = "in which programming langaguage code is written?"

response = rag_chain.invoke(question)

print("\n--- AI RESPONSE ---")
print(response)