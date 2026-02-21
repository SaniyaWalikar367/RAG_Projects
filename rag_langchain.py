from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os, getpass
os.environ["HF_TOKEN"] = getpass.getpass('Huggingface Token:')


plain_texts = [
    "LangChain helps developers build RAG applications.",
    "RAG stands for Retrival Augmented Generation.",
    "FAISS is used for vector similarity search.",
]

documents=[Document(page_content=text)for text in plain_texts]
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(documents,embeddings)
retriever = vectorstore.as_retriever()
llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

# Define a prompt template
template = """Given the following context and a question, generate a concise answer based solely on the provided context. If the answer cannot be found in the context, respond with "I don't know."

Context:
{context}

Question: {question}

Concise Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrv_chain = retriever | format_docs

rag_chain = (
    {"context": retrv_chain, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query = "What does RAG stand for?"
result = rag_chain.invoke(query)

print("Answer:\n", result)
     