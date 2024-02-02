import numpy as np
import torch as torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('../')
from embedding import *

# Load the database
restaurant_df = pd.read_csv('restaurant_embeddings_full.csv')
dish_df = pd.read_csv('food_embeddings_full.csv')

# Encode Query
def encode_query(query):
    query_embedding = encode(query)
    return query_embedding

# Compute Cosine Similarity between query and restaurant embeddings
def compute_cosine_similarities_restaurants(query_embedding, restaurant_df):
    cosine_similarities = []
    for restaurant in restaurant_df['BERT embeddings']:
        array_elements = restaurant.strip("[]").split(", ")
        array_list = [float(element) for element in array_elements]
        restaurant = torch.tensor(array_list)
        cosine_similarities.append(cosine_similarity(query_embedding, restaurant))
    Indexes = restaurant_df['index'].tolist()
    restaurant_score_pairs = list(zip(Indexes, cosine_similarities))
    restaurant_score_pairs = sorted(restaurant_score_pairs, key=lambda x: x[1], reverse=True)
    return restaurant_score_pairs

# Compute Cosine Similarity between query and restaurant suggestion menu items
def compute_cosine_similarities_dishes(query_embedding, rest_suggestion, dish_df):
    filtered_df = dish_df[dish_df['Restaurant Name'].str.contains(rest_suggestion)]
    row_indexes = filtered_df.index.tolist()
    dish_embeddings_str = dish_df["BERT embeddings"]
    dish_embeddings = []
    for dish_str in dish_embeddings_str:
        array_elements = dish_str.strip("[]").split(", ")
        array_list = [float(element) for element in array_elements]
        dish_embeddings.append(torch.tensor(array_list))
    dish_indexes = dish_embeddings[row_indexes[0]:(row_indexes[-1]+1)]
    cosine_similarities = []
    for dish in dish_indexes:
        cosine_similarities.append(cosine_similarity(query_embedding, dish))
    Indexes = filtered_df['Index'].tolist()
    dish_score_pairs = list(zip(Indexes, cosine_similarities))
    dish_score_pairs = sorted(dish_score_pairs, key=lambda x: x[1], reverse=True)
    return dish_score_pairs

# Get restaurant suggestion
def get_restaurant_suggestion(restaurant_df, restaurant_score_pairs):
    rest_suggestion = restaurant_df.iloc[restaurant_score_pairs[0][0]]['Restaurant Name']
    return rest_suggestion

# Get dish suggestion
def get_dish_suggestion(dish_df, dish_score_pairs):
    dish_suggestion = dish_df.iloc[dish_score_pairs[0][0]]['Menu Item']
    return dish_suggestion

# Print suggested restaurant and dish name
def print_suggestions(rest_suggestion, dish_suggestion):
    print_string = 'I suggest going to ' + rest_suggestion + ' and ordering the ' + dish_suggestion + ' dish. Enjoy!'
    print(print_string)

# Main function
def main():
    query = 'start'
    while(query):
        query = input("What kind of food are you looking for? : ")
        if query == '':
            break
        query_embedding = encode_query(query)
        restaurant_score_pairs = compute_cosine_similarities_restaurants(query_embedding, restaurant_df)
        rest_suggestion = get_restaurant_suggestion(restaurant_df, restaurant_score_pairs)
        dish_score_pairs = compute_cosine_similarities_dishes(query_embedding, rest_suggestion, dish_df)
        dish_suggestion = get_dish_suggestion(dish_df, dish_score_pairs)
        print_suggestions(rest_suggestion, dish_suggestion)

# Run the main function
if __name__ == "__main__":
    main()
