from  sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import faiss
import os ,getpass
from dotenv import load_dotenv
os.environ['HUGGINGFACE_TOKEN'] = getpass.getpass('Huggingface Token:')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

get_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

documents =[
    "A Christmas Carol is a novella by Charles Dickens, first published in London on 19 December 1843.",
    "The story tells of sour and stingy Ebenezer Scrooge's ideological, ethical, and emotional transformation after \n"
    "the supernatural visits of Jacob Marley and the Ghosts of Christmas Past, Present, and Yet to Come.",
    "The novella met with instant success and critical acclaim. It is regarded as one of the greatest Christmas stories ever written."
]
query = "What is the significance of Christmas Eve in A Christmas Carol?"

query_embedding = get_embeddings.encode([query])
document_embeddings = get_embeddings.encode(documents)

print("Document Embeddings Shape:", document_embeddings.shape)
print("Query Embeddings Shape:", query_embedding.shape)