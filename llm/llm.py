from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from retrieval.reranker import advance_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

Google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  temperature = 0.1,
  max_retries= 2
)

## this would create a question bassed on history

standalone_system_prompt = (
  "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

standalone_question_prompt = ChatPromptTemplate.from_messages([
  ("system",standalone_system_prompt),
  MessagesPlaceholder("chat_history"),
  ("human","{input}")
])

history_aware_retriever = create_history_aware_retriever(
  llm,advance_retriever,standalone_question_prompt
)

## answering prompt

system_prompt = (
    "You are a Senior Engineer. Answer using ONLY the following code snippets. "
    "If you can't find the answer, say 'Logic not found.'\n\n"
    "CONTEXT:\n{context}"
)

main_prompt = ChatPromptTemplate.from_messages([
  ("system",system_prompt),
  MessagesPlaceholder("chat_history"),
  ("human","{input}")
])

question_answer_chain = create_stuff_documents_chain(llm,main_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

chat_history = []

def ask_question(user_input):
  global chat_history
  response = rag_chain.invoke({"input":user_input , "chat_history":chat_history})

  chat_history.extend([
    HumanMessage(content=user_input),
    AIMessage(content=response["answer"])
  ])

  return response["answer"]


print(ask_question("In which language is this written?"))
print(ask_question("Can you explain its main function?"))



## withou memory llm
# template = """
# You are a Senior Engineer. Answer using ONLY the following code snippets. 
# If you can't find the answer, say "Logic not found."

# CONTEXT:
# {context}

# QUESTION:
# {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# def format_docs(docs):
#     """
#     Takes the raw code snippets and turns them into 
#     one clean string for the AI to read.
#     """
#     formatted_list = []
#     for d in docs:
#         source_file = d.metadata.get('source')
#         code_content = d.page_content
#         block = f"--- SOURCE FILE: {source_file} ---\n{code_content}"
#         formatted_list.append(block)
    
#     return "\n\n".join(formatted_list)

# rag_chain = (
#   {"context": advance_retriever | format_docs, "question" :RunnablePassthrough()}
#   | prompt
#   | llm
#   | StrOutputParser()
# )

# question = "in which programming langaguage code is written?"

# response = rag_chain.invoke(question)

# print("\n--- AI RESPONSE ---")
# print(response)