# embedding.py
# functions: encode, mean_pooling, max_pooling,cosine_similarity, euclidean_distance


import numpy as np
import torch as torch 
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Load the model
# Hugging Face Models to try
# paraphrase-multilingual-MiniLM-L12-v2
# msmarco-distilbert-cos-v5
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")

# referenced - https://huggingface.co/sentence-transformers/msmarco-distilbert-cos-v5
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Compute the sum of token embeddings weighted by attention mask
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    # Compute the average of token embeddings by dividing the sum by the total attention mask values
    avg_embeddings = sum_embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return avg_embeddings
    
    
    
# Not currently in use - can be used instead of mean_pooling needs more research
def max_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Set masked embeddings to a very small value so they do not affect the max pooling operation
    masked_embeddings = token_embeddings + (1 - input_mask_expanded) * float('-inf')

    # Perform max pooling along the token embeddings axis
    max_pooled_embeddings, _ = torch.max(masked_embeddings, 1)

    return max_pooled_embeddings
    
    
    
def encode(texts):
    # Tokenize sentences - AutoTokenizer
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = normalize(embeddings, p=2, dim=1)
    
    return embeddings
    
    
#  needs testing
def normalize(input, p=2, dim=1, eps=1e-12, out=None):

    # Compute the norm of the input tensor along the specified dimension
    norm = input.norm(p, dim, keepdim=True).clamp(min=eps)

    # Divide the input tensor by the norm to obtain the normalized tensor
    if out is None:
        # If out is not specified, create a new tensor for the result
        result = input / norm
    else:
        # If out is specified, fill it with the normalized values
        torch.div(input, norm, out=out)

    return result
    
    
    
    
# assumes 2 vectors have the same dimensions 
def cosine_similarity(A , B ):
  # cosine similarity 
  return np.dot(A,B)/(np.linalg.norm(A) * np.linalg.norm(B))



def euclidean_distance(A, B):
    assert len(A) == len(B), "Vectors A and B must have the same dimensions."

    squared_diff = np.power(A - B, 2)
    sum_squared_diff = np.sum(squared_diff)
    euclidean_dist = np.sqrt(sum_squared_diff)

    return euclidean_dist

