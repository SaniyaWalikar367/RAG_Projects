# RAG with LangChain, FAISS & Hugging Face

A simple **Retrieval-Augmented Generation (RAG)** project built using LangChain, FAISS and the Hugging Face Inference API.

This project demonstrates how to retrieve relevant documents using vector similarity search and generate answers using an LLM based on the retrieved context.

---

## Features

- Text embedding using Hugging Face embeddings
- Vector similarity search using FAISS
- Retrieval-Augmented Generation (RAG) pipeline
- Simple in-memory chat history
- Runs using Hugging Face Inference API (no local LLM required)

---

## Tech Stack

- LangChain – LLM orchestration framework  
- FAISS – Vector similarity search library  
- Hugging Face – Model hosting & inference  
- Python

---

## Project Structure

1.create virtual environment
-python -m venv RAG_Projects

Activate it
-RAG_Projects\Scripts\activate

 ## requirements.txt
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-text-splitters
sentence-transformers
faiss-cpu
huggingface-hub
python-dotenv

2.install dependencies
-pip install -r requirements.txt

3.Hugging Face Token
tThis project uses the stanadard environment variable:"HUGGINGFACEHUB_API_TOKEN"

## Run the Project
 python RAG_langchain.py

## Author
Saniya Walikar
