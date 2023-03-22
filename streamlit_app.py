import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
import os
import requests
from io import BytesIO

# Add a title
st.title('Postcards visual search')
# Add some text
st.text('Search based on visual understanding')

img_names = []
with open('img_ids.txt') as in_file:
    for line in in_file:
        line = line.strip()
        img_names.append(line)

emb_matrix = torch.load('clip_emb_matrix.pt', map_location=lambda storage, loc: storage)

model_multi = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Next, we define a search function.
def search(query, k=10):
    allIm = []
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model_multi.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, emb_matrix, top_k=k)[0]
    
    for hit in hits:
        elList = img_names[hit['corpus_id']].split('/')
        url = 'http://www.ccl.kuleuven.be/varia/hackathon/main/thumbnails/' + elList[8]
        print(url)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        allIm.append(img)
    return allIm

search_query = st.text_input('Search for:', 'beautiful nature')
res = search(search_query)
st.image(res, width=500)
