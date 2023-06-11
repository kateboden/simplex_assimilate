# import langchain
# load data from data/
# split the text into chunks
# use open ai to embed the chunks
# create a faiss index

import langchain
import numpy as np
import faiss
import os
import pickle
import json

def load_data():
    with open('data/combined.txt', 'r') as f:
        text = f.read()
    return text

def split_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def embed_chunks(chunks):
    embedder = langchain.Embedder()
    return embedder.embed(chunks)

def create_index(embeddings):
    index = faiss.IndexFlatIP(768)
    index.add(embeddings)
    return index

def save_index(index):
    faiss.write_index(index, 'data/index.faiss')

def load_index():
    return faiss.read_index('data/index.faiss')

# one of the best python libraries for developing command line interfaces
# is fire by google. It's a single decorator that turns any function into
# a command line interface. It's really cool.